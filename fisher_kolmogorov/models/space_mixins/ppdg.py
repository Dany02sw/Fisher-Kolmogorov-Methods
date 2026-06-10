from dolfin import *
import ufl

from fisher_kolmogorov.utilities.enum_utilities      import SpaceMethod
from fisher_kolmogorov.utilities.transform_utilities import Exponential
from fisher_kolmogorov.utilities.fenics_utilities    import havg

class SpaceMixinPpDg:
    """Positivity-preserving DG (PPDG) space discretization mixin."""

    def _init_space(self, eps, eta_0, smoothing=0.0):
        self.eps   = eps   if isinstance(eps,   ufl.core.expr.Expr) else Constant(eps)
        self.eta_0 = eta_0 if isinstance(eta_0, ufl.core.expr.Expr) else Constant(eta_0)
        self.SM    = SpaceMethod.PPDG
        self.T     = Exponential(eps=smoothing)

    def _BuildFunctionSpaces(self, l=1):
        l = int(l)
        super()._BuildFunctionSpaces(l)
        self.R  = VectorFunctionSpace(self.mesh, "DG", l)
        self.WR = self.W
        self.l  = l

    def _BuildSpatialForm(self):

        # Data
        D     = self.D
        alpha = self.alpha

        # Geometry
        self.n    = FacetNormal(self.mesh)
        h         = CellDiameter(self.mesh)

        # Penalty
        self.zeta = self.eta_0*avg(tr(D))*avg(self.l*self.l)/havg(h)
        eps  = self.eps
        tau  = self.tau

        # Measures
        dx, dS, ds = self.dx, self.dS, self.ds

        def A(u_transf, v, w, eta_u):
            return u_transf*inner(dot(D, grad(v)), grad(w))*dx \
                + eta_u*inner(jump(v, self.n), jump(w, self.n))*dS \
                - inner(avg(u_transf*dot(D, grad(v))), jump(w, self.n))*dS \
                - inner(jump(v, self.n), avg(u_transf*dot(D, grad(w))))*dS

        def F_space(components_now, components_time, transf_time, Force, gN):
            (lamb,)   = components_now
            (lamb_t,) = components_time
            phi       = TestFunction(self.WR)

            max2 = ufl.Max(self.T(lamb)('+'), self.T(lamb)('-'))**2
            eta  = self.zeta*max2*ufl.Max(self.T(abs(lamb)('+')), self.T(abs(lamb)('-')))

            lamb_o   = self.U_old if hasattr(self, 'U_old') else lamb
            tht      = self.tht   if hasattr(self, 'tht')   else Constant(1.0)
            max2_old = ufl.Max(self.T(lamb_o)('+'), self.T(lamb_o)('-'))**2
            eta_old  = self.zeta*max2_old*ufl.Max(self.T(abs(lamb_o)('+')), self.T(abs(lamb_o)('-')))

            F = - alpha*transf_time*(1.0 - transf_time)*phi*dx \
                + (eps/tau)*lamb*phi*dx \
                + (eps/tau)*inner(dot(D, grad(lamb)), grad(phi))*dx \
                + (eps/tau)*inner(self.zeta*jump(lamb, self.n), jump(phi, self.n))*dS \
                + tht*A(self.T(lamb), lamb, phi, eta) + (1.0 - tht)*A(self.T(lamb_o), lamb_o, phi, eta_old) \
                - inner(Force, phi)*dx \
                + inner(gN, self.n)*phi*ds

            return F, lamb, phi

        return F_space

    def _SetInitialCondition(self, c_0):
        assign(self.U, project(self.T.inv(c_0), self.W))

    def _SolvePostprocessing(self, t_val):
        # Extract solution
        lamb_h = self.U
        c_h    = project(self.T(lamb_h), self.W)

        # Compute min and max values
        c_min = c_h.vector().min()
        c_max = c_h.vector().max()

        # Print the bounds for both the variables
        print(f"{'─'*80}\n")
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
        print(f"  c_h  ∈ [{c_min: 7.6f}, {c_max: 7.6f}]   ‖c_ex  − exp(lamb_h)‖_L²   = {E_L2:.4e}")
        print(f"  {'':30}  ‖c_ex  − exp(lamb_h)‖_DG   = {E_DG:.4e}")
        print(f"{'─'*80}\n")

        return E_L2, E_DG, lamb_h, c_h