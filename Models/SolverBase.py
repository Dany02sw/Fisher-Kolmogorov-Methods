from dolfin import *
import ufl

class SolverBase:
    def __init__(self, mesh, D, alpha, c_0):
        self.mesh  = mesh
        self.dx    = Measure("dx", domain=mesh)
        self.dS    = Measure("dS", domain=mesh)
        self.ds    = Measure("ds", domain=mesh)
        self.D     = as_tensor(D)
        self.alpha = alpha if isinstance(alpha, ufl.core.expr.Expr) else Constant(alpha)
        self.c_0   = c_0  # This will be used as exact solution if run with ConvergenceTest
        self.WR    = None # Just for clarity, it will be filled in the derived classes
        self.SM    = None
        self.TM    = None
    
    def _BuildFunctionSpaces(self, l):
        raise NotImplementedError
    
    def _BuildFunctions(self):
        raise NotImplementedError
    
    def _BuildSpatialForm(self, tau, tht):
        raise NotImplementedError

    def _BuildTimeForm(self):
        raise NotImplementedError
    
    def _BuildVariationalForms(self, tau, tht):
        F_space   = self._BuildSpatialForm(self, ...)
        F_time    = self._BuildTimeForm(self, ...)
        self.Form = F_space + F_time

    def _SetSourceTerm(self, x, t, extForce, NeumannBC):
        self.Force = extForce(x, t)  if extForce  else Constant(0.0)
        self.gN    = NeumannBC(x, t) if NeumannBC else Constant((0.0, 0.0))

    def _SeiInitialCondition(self):
        raise NotImplementedError
    
    def _BuildNonlinearSolver(self, tol=1e-8, maxIt=200):
        # Jacobian
        dU = TrialFunction(self.WR)
        J  = derivative(self.Form, self.U, dU)

        # Nonlinear solver
        problem      = NonlinearVariationalProblem(self.Form, self.U, J=J)
        self.solver  = NonlinearVariationalSolver(problem)
        prm          = self.solver.parameters

        prm['nonlinear_solver']                  = 'snes'
        prm['snes_solver']['method']             = 'newtonls'
        prm['snes_solver']['line_search']        = 'bt'
        prm['snes_solver']['absolute_tolerance'] = tol
        prm['snes_solver']['relative_tolerance'] = tol
        prm['snes_solver']['maximum_iterations'] = maxIt
        prm['snes_solver']['linear_solver']      = 'lu'
        prm['snes_solver']['preconditioner']     = 'none'

    def _UpdateOldState(self):
        raise NotImplementedError
    
    def _ValidateInput(self, t0, dt, T, l):
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        if T <= t0:
            raise ValueError(f"Final time T ({T}) must be greater than initial time t0 ({t0})")
        if not isinstance(l, int) or l < 0:
            raise ValueError(f"Polynomial degree l must be a non-negative integer, got {l}")
    
    def _PrintSolverInfo(self):
        raise NotImplementedError
    
    def _PrintSolveIteration(self):
        raise NotImplementedError
    
    def _PrintConvergenceIteration(self):
        raise NotImplementedError
    
    def Solve(self):
        raise NotImplementedError
    
    def ConvergenceTest(self):
        raise NotImplementedError 