from Models.SolverTheta import SolverTheta

from dolfin import *
import ufl

from Utilities.EnumUtilities      import SpaceMethod
from Utilities.TransformUtilities import Identity
from Utilities.FEniCSUtilities    import havg

class SolverDgTheta(SolverTheta):
    def __init__(self, mesh, D, alpha, c_0, eta_0):
        super().__init__(mesh, D, alpha, c_0, transform=Identity())
        self.eta_0   = eta_0   if isinstance(eta_0,   ufl.core.expr.Expr) else Constant(eta_0)
        self.SM      = SpaceMethod.DG

    def _BuildFunctionSpaces(self, l=1):
        super()._BuildFunctionSpaces(l)
        self.R  = VectorFunctionSpace(self.mesh, "DG", l)
        self.WR = self.W
        self.l  = l

    # def _BuildSpatialForm(self):
    #     # Data
    #     D     = self.D
    #     alpha = self.alpha
    #     tht   = self.tht

    #     # Geometry
    #     self.n    = FacetNormal(self.mesh)
    #     h         = CellDiameter(self.mesh)

    #     # Extract functions
    #     c     = self.U
    #     w     = TestFunction(self.WR)
    #     c_old = self.U_old

    #     # Force term and Neumann BC
    #     Force     = self.Force
    #     Force_old = self.Force_old
    #     gN        = self.gN      
    #     gN_old    = self.gN_old 

    #     # Measures
    #     dx = self.dx
    #     dS = self.dS
    #     ds = self.ds

    #     # Penalty coefficient
    #     self.eta = self.eta_0 * self.l*self.l / havg(h)

    #     # A form (only on internal facets)
    #     A = lambda u, v: inner(D*grad(u), grad(v))*dx \
    #         + self.eta*inner(jump(u, self.n), jump(v, self.n))*dS \
    #         - inner( avg(dot(D, grad(u))), jump(v, self.n) )*dS \
    #         - inner( jump(u, self.n), avg(dot(D, grad(v))) )*dS

    #     # Form
    #     F = A(tht*c + (1.0 - tht)*c_old, w) \
    #         - alpha*(tht*c + (1.0 - tht)*c_old)*(1.0 - (tht*c + (1.0 - tht)*c_old))*w*dx \
    #         - tht*Force*w*dx - (1.0 - tht)*Force_old*w*dx \
    #         + tht*inner(gN, self.n)*w*ds + (1.0 - tht)*inner(gN_old, self.n)*w*ds
        
    #     return F, c, w
    
    def _BuildSpatialForm(self):
        D     = self.D
        alpha = self.alpha
        self.n   = FacetNormal(self.mesh)
        h        = CellDiameter(self.mesh)
        self.eta = self.eta_0 * self.l*self.l / havg(h)
        dx, dS, ds = self.dx, self.dS, self.ds

        def A(u, v):
            return inner(D*grad(u), grad(v))*dx \
                + self.eta*inner(jump(u, self.n), jump(v, self.n))*dS \
                - inner(avg(dot(D, grad(u))), jump(v, self.n))*dS \
                - inner(jump(u, self.n), avg(dot(D, grad(v))))*dS

        def F_space(components_now, components_time, Force, gN):
            (c,)  = components_now
            (c_t,) = components_time
            w     = TestFunction(self.WR)
            F = A(c_t, w) \
                - alpha*c_t*(1.0 - c_t)*w*dx \
                - inner(Force, w)*dx \
                + inner(gN, self.n)*w*ds
            return F, c, w

        return F_space
    
    def _SetInitialCondition(self, c_0):
        assign(self.U, project(c_0, self.W))

    def _SolvePostprocessing(self, t_val):
        # Extract solution
        c_h = self.U

        # Compute min and max values
        c_min     = c_h.vector().min()
        c_max     = c_h.vector().max()

        # Print the bounds for both the variables
        print(f"  t = {t_val:.{self.decimals}f}")
        print(f"{'-'*80}")
        print(f"  c_h      ∈ [{c_min: 7.6f}, {c_max: 7.6f}]")
        print(f"{'─'*80}\n")

        return c_h
    
    def _ConvergenceTestPostprocessing(self, t_val):
        # Extract solution
        c_h = self.U

        # Compute errors
        c_err = self.c_ex - c_h
        E_L2  = sqrt(assemble(c_err*c_err*self.dx))
        E_DG  = sqrt(assemble(
            inner(dot(self.D, grad(c_err)), grad(c_err))*self.dx + self.eta*inner(jump(c_err, self.n), jump(c_err, self.n))*self.dS
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

        return E_L2, E_DG