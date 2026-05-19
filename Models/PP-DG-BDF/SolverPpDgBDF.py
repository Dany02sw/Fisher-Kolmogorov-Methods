from Models.SolverBDF import SolverBDF

from dolfin import *
import ufl

from Utilities.EnumUtilities      import SpaceMethod
from Utilities.TransformUtilities import Exponential
from Utilities.FEniCSUtilities    import havg

class SolverPpDgBDF(SolverBDF):
    def __init__(self, mesh, D, alpha, c_0, eps, eta_0, smoothing=0.0):
        super().__init__(mesh, D, alpha, c_0, transform=Exponential(eps=smoothing))
        self.eps     = eps     if isinstance(eps,     ufl.core.expr.Expr) else Constant(eps)
        self.eta_0   = eta_0   if isinstance(eta_0,   ufl.core.expr.Expr) else Constant(eta_0)
        self.SM      = SpaceMethod.PPDG

    def _BuildFunctionSpaces(self, l=1):
        super()._BuildFunctionSpaces(l)
        self.R        = VectorFunctionSpace(self.mesh, "DG", l)
        self.WR = self.W
        self.l  = l

    def _BuildSpatialForm(self):
        # Data
        D     = self.D
        alpha = self.alpha

        # Geometry
        self.n    = FacetNormal(self.mesh)
        h         = CellDiameter(self.mesh)
        self.zeta = self.eta_0*avg(tr(D))*avg(self.l*self.l)/havg(h)

        # Extract functions
        lamb     = self.U
        phi      = TestFunction(self.WR)

        # Force term and Neumann BC
        Force     = self.Force
        gN        = self.gN

        # Measures
        dx = self.dx
        dS = self.dS
        ds = self.ds

        # Penalty coefficient
        max2      = ufl.Max(self.T(lamb)('+'), self.T(lamb)('-'))**2
        eta       = self.zeta*max2*ufl.Max(self.T(abs(lamb)('+')), self.T(abs(lamb)('-')))

        # A form (only on internal facets)
        A = lambda u, v, w, eta_u: self.T(u)*inner(dot(D, grad(v)), grad(w))*dx \
            + eta_u*inner(jump(v, self.n), jump(w, self.n))*dS \
            - inner( avg(self.T(u)*dot(D, grad(v))), jump(w, self.n) )*dS \
            - inner( jump(v, self.n), avg(self.T(u)*dot(D, grad(w))) )*dS

        # Form
        F = - alpha*self.T(lamb)*(1.0 - self.T(lamb))*phi*dx \
            + (self.eps/self.tau)*lamb*phi*dx \
            + (self.eps/self.tau)*inner(dot(D, grad(lamb)), grad(phi))*dx \
            + (self.eps/self.tau)*inner(self.zeta*jump(lamb, self.n), jump(phi, self.n))*dS \
            + A(lamb, lamb, phi, eta) \
            - Force*phi*dx \
            + inner(gN, self.n)*phi*ds
        
        return F, lamb, phi
    
    def _SetInitialCondition(self, c_0):
        assign(self.U, project(self.T.inv(c_0), self.W))

    def _SolvePostprocessing(self, t_val):
        # Extract solution
        lamb_h = self.U
        c_h    = project(self.T(lamb_h), self.W)

        # Compute min and max values
        c_min  = c_h.vector().min()
        c_max  = c_h.vector().max()

        # Print the bounds for both the variables
        print(f"  t = {t_val:.{self.decimals}f}")
        print(f"{'-'*80}")
        print(f"  c_h      ∈ [{c_min: 7.6f}, {c_max: 7.6f}]")
        print(f"{'─'*80}\n")

        return c_h, lamb_h
    
    def _ConvergenceTestPostprocessing(self, t_val):
        # Extract solution
        lamb_h = self.U
        c_h    = project(self.T(lamb_h), self.W)

        # Compute errors
        c_err = self.c_ex - self.T(lamb_h)
        E_L2  = sqrt(assemble(c_err*c_err*self.dx))
        E_DG  = sqrt(assemble(
            inner(dot(self.D, grad(c_err)), grad(c_err))*self.dx + self.zeta*inner(jump(c_err, self.n), jump(c_err, self.n))*self.dS
        ))

        # Compute min and max values
        c_min = c_h.vector().min()
        c_max = c_h.vector().max()

        # Print the bounds and the errors for both the variables
        print(f"\n{'─'*80}")
        print(f"  t = {t_val:.{self.decimals}f}")
        print(f"{'-'*80}")
        print(f"  c_h  ∈ [{c_min: 7.6f}, {c_max: 7.6f}]   ‖c_ex  − c_h‖_L²   = {E_L2:.4e}")
        print(f"  {'':30}  ‖c_ex  − c_h‖_DG   = {E_DG:.4e}")
        print(f"{'─'*80}\n")

        return E_L2, E_DG, lamb_h