from Models.SolverBase import SolverBase

from dolfin import *
from pathlib import Path

from Utilities.FEniCSUtilities import Normalize
from Utilities.ClassUtilities import OutputManager

BDF_COEFFS = {
    1: {"beta": 1.0,          "a": [1.0]                                                                          },
    2: {"beta": 2.0 / 3.0,    "a": [4.0/3.0    ,     -1.0/3.0]                                                    },
    3: {"beta": 6.0 / 11.0,   "a": [18.0/11.0  ,    -9.0/11.0,    2.0/11.0]                                       },
    4: {"beta": 12.0 / 25.0,  "a": [48.0/25.0  ,   -36.0/25.0,   16.0/25.0,    -3.0/25.0]                         },
    5: {"beta": 60.0 / 137.0, "a": [300.0/137.0, -300.0/137.0, 200.0/137.0,  -75.0/137.0, 12.0/137.0]             },
    6: {"beta": 60.0 / 147.0, "a": [360.0/147.0, -450.0/147.0, 400.0/147.0, -225.0/147.0, 72.0/147.0, -10.0/147.0]},
}


class SolverBDF(SolverBase):
    def __init__(self, mesh, D, alpha, c_0):
        super.__init__(mesh, D, alpha, c_0)
        self.W = None
        
    def _BuildFunctions(self, nu):
        self.U     = Function(self.WR)
        self.u_old = [Function(self.W) for _ in range(1, nu)] # small letter because when you have mixed spaces the backward steps will only use the first space

    def _BuildTimeForm(self, tau, nu, u, v, u_old, transform=lambda x: x):
        dx = Measure("dx", self.mesh)

        # Import coefficients
        beta = BDF_COEFFS[nu]["beta"]
        a    = BDF_COEFFS[nu]["a"]

        # Compute the linear combination of backward steps
        transformed_old_steps = sum(a[j]*transform(u_old[j]) for j in range(len(a)))

        return (1.0/(tau*beta))*(transform(u) - transformed_old_steps)*v*dx
    
    def _UpdateOldState(self, nu):
        u = self.U.sub(0)
        for j in range(nu-1, 0, -1):
            self.u_old[j].assign(self.u_old[j-1])
        self.u_old[0].assign(u)

    def _ValidateInput(self, t0, dt, nu, T, l):
        super()._ValidateInput(t0, dt, T, l)
        if not isinstance(nu, int) or nu < 1 or nu > 6:
            raise ValueError(f"Values of nu must be an integer from 1 to 6, got {nu}")
        
    def Solve(self, t0, dt, T, nu, l, tol, maxIt, extForce=None, NeumannBC=None):
  
        self._ValidateInput(t0, dt, T, nu, l)
        
        # Mesh data
        x = SpatialCoordinate(self.mesh)

        # Time loop parameters
        t_val  = t0 
        nsteps = round((T - t0)/dt)
        t      = Constant(t0)

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Force and Neumann BC
        self._SetSourceTerm(x, t, extForce, NeumannBC)

        # Initial guess for solver and old terms
        c_0 = self.c_0(x, t)
        c_0 = project(c_0, self.W)
        c_0 = Normalize(c_0)
        self._SeiInitialCondition(c_0)
        self._UpdateOldState()

        # Variational form and solver
        self._BuildVariationalForms(dt, nu)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize output manager
        base_results_path = Path(__file__).parent / "results"
        exporter = OutputManager(self.mesh, base_results_path)
        exporter.open()

        # Save initial condition
        exporter.save(c_0, t_val)

        # Time loop
        for i in range(nsteps):
            # Update time step
            t_val += dt
            t.assign(t_val)

            # Solve the problem
            self.solver.solve()

            # Update old solutions
            self._UpdateOldState()

            # Print the iteration
            c_h = self._PrintSolveIteration()

            # Save solution
            exporter.save(c_h, t_val)

        # Close output file
        exporter.close()

    # --- Method for a convergence test using c_0 as exact solution ---
    def ConvergenceTest(self, t0, dt, T, nu, l, tol, maxIt):

        self._ValidateInput(t0, dt, T, nu, l)

        # Mesh data
        x     = SpatialCoordinate(self.mesh)
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0

        # Time loop parameters
        t_val  = t0 
        nsteps = round((T - t0)/dt)
        t      = Constant(t0)

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
        self._SeiInitialCondition(self.c_ex)
        self._UpdateOldState()

        # Forms ansd solver
        self._BuildVariationalForms(dt, nu)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize error lists
        E_c    = None
        E_grad = None

        # Time loop
        for i in range(nsteps):

            # Update time step
            t_val += dt
            t.assign(t_val)

            # Solve the problem
            self.solver.solve()

            # Update old solutions
            self._UpdateOldState()

            # Print convergence iterations
            E_c, E_grad = self._PrintConvergenceIteration()

        return E_c, E_grad, h_avg