from Models.SolverBase import SolverBase

from dolfin import *
from pathlib import Path

from Utilities.FEniCSUtilities import Normalize
from Utilities.IOUtilities import OutputManager
from Utilities.MathUtilities import get_decimals

class SolverTheta(SolverBase):
    def __init__(self, mesh, D, alpha, c_0, transform):
        super().__init__(mesh, D, alpha, c_0, transform)
        self.W         = None
        self.R         = None
        self.Force     = None
        self.Force_old = None
        self.gN        = None
        self.gN_old    = None
        
    def _BuildFunctions(self):
        self.U     = Function(self.WR)
        self.U_old = Function(self.WR)

    def _BuildTimeForm(self, tau, u, v):
        dx    = self.dx
        u_old = split(self.U_old)[0]
        return (1.0/tau)*(self.T(u) - self.T(u_old))*v*dx
    
    def _SetSourceTerm(self, x, t, extForce, NeumannBC):
        super()._SetSourceTerm(x, t, extForce, NeumannBC)
        self.Force_old = Function(self.W)
        self.gN_old    = Function(self.R)
    
    def _UpdateOldState(self): # <---- OVERRIDE INSIDE PPG, WHICH DOES NOT HAVE AN R SPACE
        self.U_old.assign(self.U)
        self.Force_old.assign(project(self.Force, self.W))
        self.gN_old.assign(project(self.gN, self.R))
    
    def _ValidateInput(self, t0, dt, tht, T, l):
        super()._ValidateInput(t0, dt, T, l)
        if not (0.0 <= tht <= 1.0):
            raise ValueError("tht must be between in [0, 1]")


    def Solve(self, t0, dt, T, tht, l, tol, maxIt, extForce=None, NeumannBC=None):
  
        self._ValidateInput(t0, dt, tht, T, l)
        
        # Mesh data
        x = SpatialCoordinate(self.mesh)

        # Time loop parameters
        t_val    = t0 
        nsteps   = round((T - t0)/dt)
        t        = Constant(t0)
        self.tht = Constant(tht)

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Force and Neumann BC
        self._SetSourceTerm(x, t, extForce, NeumannBC)

        # Initial guess for solver and old terms
        c_0 = self.c_0(x, t)
        c_0 = project(c_0, self.W)
        c_0 = Normalize(c_0)
        self._SetInitialCondition(c_0)
        self._UpdateOldState()

        # Variational form and solver
        self._BuildVariationalForms(dt)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize output manager
        base_results_path = Path(__file__).parent / "results"
        exporter = OutputManager(self.mesh, base_results_path)
        exporter.open()

        # Save initial condition
        exporter.save(c_0, t_val)

        # Time loop
        self.decimals = get_decimals(dt)
        for i in range(nsteps):
            # Update time step
            t_val += dt
            t.assign(t_val)

            # Solve the problem
            self.solver.solve()

            # Print the iteration
            c_h = self._SolvePostprocessing(t_val)
            
            # Update old solutions
            self._UpdateOldState()

            # Save solution
            exporter.save(c_h, t_val)

        # Close output file
        exporter.close()


    def ConvergenceTest(self, t0, dt, T, tht, l, tol, maxIt):

        self._ValidateInput(t0, dt, tht, T, l)

        # Mesh data
        x     = SpatialCoordinate(self.mesh)
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0

        # Time loop parameters
        t_val    = t0 
        nsteps   = round((T - t0)/dt)
        t        = Constant(t0)
        self.tht = Constant(tht)

        # Data
        D         = self.D
        alpha     = self.alpha
        self.c_ex = self.c_0(x, t)

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Computing forcing term
        c_t        = diff(self.c_ex, t)
        Delta_c    = div(D*grad(self.c_ex))
        self.Force = c_t - Delta_c - alpha*self.c_ex*(1.0 - self.c_ex)

        # Computing Neumann BC
        self.gN = dot(D, grad(self.c_ex))

        # Initial guess for solver and old terms
        self.Force_old = Function(self.W)
        self.gN_old    = Function(self.R)
        self._SetInitialCondition(self.c_ex)
        self._UpdateOldState()

        # Forms ansd solver
        self._BuildVariationalForms(dt)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize error lists
        E_c    = None
        E_grad = None

        # Time loop
        self.decimals = get_decimals(dt)
        for i in range(nsteps):

            # Update time step
            t_val += dt
            t.assign(t_val)

            # Solve the problem
            self.solver.solve()

            # Print convergence iterations
            E_c, E_grad = self._ConvergenceTestPostprocessing(t_val)

            # Update old solutions
            self._UpdateOldState()

        return E_c, E_grad, h_avg