from Models.SolverBDF import SolverBDF

from dolfin import *
import ufl

from Utilities.EnumUtilities      import SpaceMethod
from Utilities.TransformUtilities import Identity

class SolverLdgBDF(SolverBDF):
    def __init__(self, mesh, D, alpha, c_0, C11, C12):
        super().__init__(mesh, D, alpha, c_0, transform=Identity())
        self.C11   = C11   if isinstance(C11,   ufl.core.expr.Expr) else Constant(C11)
        self.C12   = C12   if isinstance(C12,   ufl.core.expr.Expr) else Constant(C12)
        self.SM    = SpaceMethod.LDG

    def _BuildFunctionSpaces(self, l=1):
        super()._BuildFunctionSpaces(l)
        self.R        = VectorFunctionSpace(self.mesh, "DG", l)
        element_c     = self.W.ufl_element()
        element_q     = self.R.ufl_element()
        mixed_element = MixedElement([element_c, element_q])
        self.WR       = FunctionSpace(self.mesh, mixed_element)

    # def _BuildSpatialForm(self):
    #     # Geometry
    #     h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0
    #     n     = FacetNormal(self.mesh)

    #     # Data
    #     D     = self.D
    #     alpha = self.alpha
    #     C11   = self.C11
    #     C12   = self.C12*n('+')

    #     # Extract functions
    #     (c, q)  = split(self.U)
    #     Phi     = TestFunction(self.WR)
    #     (v, r)  = split(Phi)

    #     # Force term and Neumann BC
    #     Force = self.Force
    #     gN    = self.gN

    #     # Measures
    #     dx = self.dx
    #     dS = self.dS
    #     ds = self.ds

    #     # Forms
    #     Fq = inner(q, r)*dx \
    #         + inner(c, div(dot(D.T, r)))*dx \
    #         - ( avg(c) + inner(C12, jump(c, n)) )*jump(dot(D.T, r), n)*dS \
    #         - c*inner(dot(D.T, r), n)*ds
    #     Fc = inner(q, grad(v))*dx \
    #         - inner(avg(q) - (C11/h_avg)*jump(c, n) - C12*jump(q, n), jump(v, n))*dS \
    #         - alpha*c*(1.0 - c)*v*dx \
    #         - Force*v*dx \
    #         - inner(gN, n)*v*ds
        
    #     return Fq + Fc, c, v
    
    def _BuildSpatialForm(self):
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0
        n     = FacetNormal(self.mesh)
        D     = self.D
        alpha = self.alpha
        C11   = self.C11
        C12   = self.C12*n('+')
        dx, dS, ds = self.dx, self.dS, self.ds

        def F_space(components_now, components_time, Force, gN):
            (c, q)   = components_now
            (c_t, q_t) = components_time
            Phi      = TestFunction(self.WR)
            (v, r)   = split(Phi)

            # Fq: algebraic constraint, always at current time
            Fq = inner(q, r)*dx \
                + inner(c, div(dot(D.T, r)))*dx \
                - (avg(c) + inner(C12, jump(c, n)))*jump(dot(D.T, r), n)*dS \
                - c*inner(dot(D.T, r), n)*ds

            # Fc: differential equation, at time-discretized quantities
            Fc = inner(q_t, grad(v))*dx \
                - inner(avg(q_t) - (C11/h_avg)*jump(c_t, n) - C12*jump(q_t, n), jump(v, n))*dS \
                - alpha*c_t*(1.0 - c_t)*v*dx \
                - inner(Force, v)*dx \
                - inner(gN, n)*v*ds

            return Fq + Fc, c, v

        return F_space
    
    def _SetInitialCondition(self, c_0):
        assign(self.U.sub(0), project(c_0, self.W))
        assign(self.U.sub(1), project(dot(self.D, grad(c_0)), self.R))

    def _SolvePostprocessing(self, t_val):
        # Extract solution
        (c_h, q_h) = self.U.split(deepcopy=True)

        # Compute min and max values
        c_min = c_h.vector().min()
        c_max = c_h.vector().max()
        q_min = q_h.vector().min()
        q_max = q_h.vector().max()

        # Print the bounds for both the variables
        print(f"  t = {t_val:.{self.decimals}f}")
        print(f"{'-'*80}")
        print(f"  c_h  ∈ [{c_min: 7.6f}, {c_max: 7.6f}]")
        print(f"  q_h  ∈ [{q_min: 7.6f}, {q_max: 7.6f}]")
        print(f"{'─'*80}\n")

        return c_h, c_h
    
    def _ConvergenceTestPostprocessing(self, t_val):
        # Extract solution
        (c_h, q_h) = self.U.split(deepcopy=True)

        # Compute errors
        E_c = sqrt(assemble((self.c_ex - c_h)*(self.c_ex - c_h)*self.dx))
        E_q = sqrt(assemble(inner(dot(self.D, grad(self.c_ex)) - q_h, dot(self.D, grad(self.c_ex)) - q_h )*self.dx))

        # Compute min and max values
        c_min = c_h.vector().min()
        c_max = c_h.vector().max()
        q_min = q_h.vector().min()
        q_max = q_h.vector().max()

        # Print the bounds and the errors for both the variables
        print(f"\n{'─'*80}")
        print(f"  t = {t_val:.{self.decimals}f}")
        print(f"{'-'*80}")
        print(f"  c_h  ∈ [{c_min: 7.6f}, {c_max: 7.6f}]   ‖c_ex  − c_h‖_L²   = {E_c:.4e}")
        print(f"  q_h  ∈ [{q_min: 7.6f}, {q_max: 7.6f}]   ‖D∇c_ex − q_h‖_L²  = {E_q:.4e}")
        print(f"{'─'*80}\n")

        return E_c, E_q, c_h