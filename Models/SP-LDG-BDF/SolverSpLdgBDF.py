from Models.SolverBDF import SolverBDF

from dolfin import *
import ufl

from Utilities.EnumUtilities      import SpaceMethod
from Utilities.TransformUtilities import Sigmoid
from Utilities.SpLdgUtilities     import div_LDG, grad_LDG, inner_LDG

class SolverSpLdgBDF(SolverBDF):
    def __init__(self, mesh, D, alpha, c_0, eps, eta_0, theta, smoothing=0.0):
        super().__init__(mesh, D, alpha, c_0, transform=Sigmoid(eps=smoothing))
        self.eps     = eps     if isinstance(eps,     ufl.core.expr.Expr) else Constant(eps)
        self.eta_0   = eta_0   if isinstance(eta_0,   ufl.core.expr.Expr) else Constant(eta_0)
        self.theta   = theta   if isinstance(theta,   ufl.core.expr.Expr) else Constant(theta)
        self.SM      = SpaceMethod.SPLDG

    def _BuildFunctionSpaces(self, l=1):
        super()._BuildFunctionSpaces(l)
        self.l        = l
        self.R        = VectorFunctionSpace(self.mesh, "DG", l)
        element_c     = self.W.ufl_element()
        element_sigma = self.R.ufl_element()
        mixed_element = MixedElement([element_c, element_sigma, element_sigma, element_sigma])
        self.WR       = FunctionSpace(self.mesh, mixed_element)

    def _BuildSpatialForm(self):
        # Data
        D     = self.D
        alpha = self.alpha

        # Geometry
        n         = FacetNormal(self.mesh)
        h         = CellDiameter(self.mesh)
        f         = FacetArea(self.mesh)
        mK        = Constant(self.mesh.geometry().dim() + 1) # fine since fenics only works with simplicial elements
        param_den = ( dot(n('+'), D('+')*n('+')) + dot(n('-'), D('-')*n('-')) )
        eta_F     = self.eta_0*(self.l**2)*2.0*( ( dot(n('+'), D('+')*n('+')) )*( dot(n('-'), D('-')*n('-')) ) ) / param_den
        h_avg     = (1.0/eta_F)*( 0.5*( ( h('+')/(mK*f('+')) )**self.theta + ( h('-')/(mK*f('-')) )**self.theta ) )**(1.0/self.theta)
        gamma     = (dot(n('+'), D('+')*n('+'))) / param_den

        # Extract functions
        (w, sigma, z, r)     = split(self.U)
        Phi                  = TestFunction(self.WR)
        (psi, phi, eta, chi) = split(Phi)

        # Force term and Neumann BC
        Force = self.Force
        gN    = self.gN

        # Measures
        dx = self.dx
        dS = self.dS
        ds = self.ds

        # Forms
        F1 = inner(z, eta)*dx + grad_LDG(w, eta, n, gamma, dx, dS)
        F2 = inner(D*self.T.s2(self.T(w))*sigma, phi)*dx  - inner(D*z, phi)*dx
        F3 = inner(r, chi)*dx - inner(D*sigma, chi)*dx
        F4 = self.eps*inner_LDG(w, psi, n, gamma, h_avg, D, dx, dS, alpha) \
            + div_LDG(r, psi, n, gamma, dx, dS) + inner(gN, n)*psi*ds \
            + inner((1.0/h_avg)*jump(w, n), jump(psi, n))*dS \
            - inner(alpha*self.T(w)*(1.0 - self.T(w)), psi)*dx \
            - inner(Force, psi)*dx 
        
        return F1 + F2 + F3 + F4, w, psi
    
    def _SetInitialCondition(self, c_0):
        assign(self.U.sub(0), project(self.T.s1(c_0),           self.W))
        assign(self.U.sub(1), project(-grad(c_0),               self.R))
        assign(self.U.sub(2), project(-self.T.s2(c_0)*grad(c_0), self.R))
        assign(self.U.sub(3), project(dot(self.D, -grad(c_0)),   self.R))

    def _SolvePostprocessing(self, t_val):
        # Extract solution
        (w_h, sigma_h, z_h, r_h) = self.U.split(deepcopy=True)
        c_h                      = project(self.T(w_h), self.W)

        # Compute min and max values
        c_min     = c_h.vector().min()
        c_max     = c_h.vector().max()
        sigma_min = sigma_h.vector().min()
        sigma_max = sigma_h.vector().max()

        # Print the bounds for both the variables
        print(f"  t = {t_val:.{self.decimals}f}")
        print(f"{'-'*80}")
        print(f"  c_h      ∈ [{c_min: 7.6f}, {c_max: 7.6f}]")
        print(f"  sigma_h  ∈ [{sigma_min: 7.6f}, {sigma_max: 7.6f}]")
        print(f"{'─'*80}\n")

        return c_h, w_h
    
    def _ConvergenceTestPostprocessing(self, t_val):
        # Extract solution
        (w_h, sigma_h, z_h, r_h) = self.U.split(deepcopy=True)
        c_h                      = project(self.T(w_h), self.W)

        # Compute errors
        E_c = sqrt(assemble((self.c_ex - self.T(w_h))*(self.c_ex - self.T(w_h))*self.dx))
        E_sigma = sqrt(assemble(inner(grad(self.c_ex) + sigma_h, grad(self.c_ex) + sigma_h)*self.dx))

        # Compute min and max values
        c_min = c_h.vector().min()
        c_max = c_h.vector().max()
        sigma_max = sigma_h.vector().max()
        sigma_min = sigma_h.vector().min()

        # Print the bounds and the errors for both the variables
        print(f"\n{'─'*80}")
        print(f"  t = {t_val:.{self.decimals}f}")
        print(f"{'-'*80}")
        print(f"  {'c_h':<8} ∈ [{c_min: 7.6f}, {c_max: 7.6f}]      {'‖c_ex  − u(w_h)‖_L²':<20} = {E_c:.4e}")
        print(f"  {'sigma_h':<8} ∈ [{sigma_min: 7.6f}, {sigma_max: 7.6f}]      {'‖∇c_ex + sigma_h‖_L²':<20} = {E_sigma:.4e}")
        print(f"{'─'*80}\n")

        return E_c, E_sigma, w_h