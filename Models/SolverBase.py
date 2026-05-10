from dolfin import *
import ufl

class SolverBase:
    def __init__(self, mesh, D, alpha, c_0):
        self.mesh  = mesh
        self.D     = as_tensor(D)
        self.alpha = alpha if isinstance(alpha, ufl.core.expr.Expr) else Constant(alpha)
        self.c_0   = c_0  # This will be used as exact solution if run with ConvergenceTest
        self.WR    = None # Just for clarity, it will be filled in the derived classes
    
    def _BuildFunctionSpaces(self, l):
        raise NotImplementedError
    
    def _BuildFunctions(self):
        self.U     = Function(self.WR)
        self.U_old = Function(self.WR)
    
    def _BuildVariationalForms(self, tau, tht):
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
    
    def _ValidateInput(self, t0, dt, T, tht, l):
        raise NotImplementedError
    
    def Solve(self, ...):
        # uguale per tutti
        ...
    
    def ConvergenceTest(self, ...):
        # uguale per tutti
        ...
    
    def _PrintSolverHeader(self, ...):
        # implementazione base
        ...
    
    def _PrintConvergenceRow(self, ...):
        raise NotImplementedError