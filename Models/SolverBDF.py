from Models.SolverBase import SolverBase

from dolfin import *
from config import DEFAULT_RESULTS_DIR

from Utilities.FEniCSUtilities import Normalize
from Utilities.EnumUtilities   import TimeMethod
from Utilities.IOUtilities     import make_output_manager
from Utilities.MathUtilities   import get_decimals

BDF_COEFFS = {
    1: {"beta": 1.0,          "a": [1.0]                                                                          },
    2: {"beta": 2.0 / 3.0,    "a": [4.0/3.0    ,     -1.0/3.0]                                                    },
    3: {"beta": 6.0 / 11.0,   "a": [18.0/11.0  ,    -9.0/11.0,    2.0/11.0]                                       },
    4: {"beta": 12.0 / 25.0,  "a": [48.0/25.0  ,   -36.0/25.0,   16.0/25.0,    -3.0/25.0]                         },
    5: {"beta": 60.0 / 137.0, "a": [300.0/137.0, -300.0/137.0, 200.0/137.0,  -75.0/137.0, 12.0/137.0]             },
    6: {"beta": 60.0 / 147.0, "a": [360.0/147.0, -450.0/147.0, 400.0/147.0, -225.0/147.0, 72.0/147.0, -10.0/147.0]},
}


class SolverBDF(SolverBase):
    def __init__(self, mesh, D, alpha, c_0, transform):
        super().__init__(mesh, D, alpha, c_0, transform)
        self.TM      = TimeMethod.BDF
        self.W = None
        
    def _BuildFunctions(self):
        self.U     = Function(self.WR)
        self.u_old = [Function(self.W) for _ in range(self.nu)] # small letter because when you have mixed spaces the backward steps will only use the first space

    def _BuildTimeForm(self, tau, u, v):
        dx = self.dx

        # Import coefficients
        beta = BDF_COEFFS[self.nu]["beta"]
        a    = BDF_COEFFS[self.nu]["a"]

        # Compute the linear combination of backward steps
        transformed_old_steps = sum(a[j]*self.T(self.u_old[j]) for j in range(len(a)))

        return (1.0/(tau*beta))*(self.T(u) - transformed_old_steps)*v*dx
    
    def _BuildVariationalForms(self, tau):
        self.tau    = Constant(tau)

        # Obtain the callable
        F_space     = self._BuildSpatialForm()

        # Construct the components tuple correctly
        comps       = split(self.U) if self.WR != self.W else (self.U,)
        transf_time = self.T(comps[0])

        # Evaluate the callable in the right place
        F, u, v     = F_space(comps, comps, transf_time, self.Force, self.gN)

        # Build the time part
        F_time      = self._BuildTimeForm(tau, u, v)

        # Put space and time part together
        self.Form   = F + F_time
    
    def _UpdateOldState(self, u):
        for j in range(self.nu-1, 0, -1):
            self.u_old[j].assign(self.u_old[j-1])
        self.u_old[0].assign(u)

    def _ValidateInput(self, t0, dt, nu, T, l):
        super()._ValidateInput(t0, dt, T, l)
        if not isinstance(nu, int) or nu < 1 or nu > 6:
            raise ValueError(f"Values of nu must be an integer from 1 to 6, got {nu}")
        
    def Solve(self, t0, dt, T, nu, l, tol, maxIt, extForce=None, NeumannBC=None, output_dir=None):
  
        self._ValidateInput(t0, dt, nu ,T, l)
        
        # Mesh data
        x = SpatialCoordinate(self.mesh)

        # Time loop parameters
        t_val   = t0
        nsteps  = round((T - t0)/dt)
        t       = Constant(t0)
        self.nu = nu

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Force and Neumann BC
        self._SetSourceTerm(x, t, extForce, NeumannBC)

        # Initial guess for solver and old terms
        c_0 = project(self.c_0(x, t), self.W)
        c_0 = Normalize(c_0)
        self._SetInitialCondition(c_0)
        self.u_old[0].assign(project(self.T.inv(c_0), self.W))
        for j in range(1, nu):
            t_back = t0 - j*dt
            t.assign(t_back)
            c_back = project(self.c_0(x, t), self.W)
            c_back = Normalize(c_back)
            self.u_old[j].assign(project(self.T.inv(c_back), self.W))

        # Variational form and solver
        self._BuildVariationalForms(dt)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize output manager
        if output_dir is None:
            output_dir = DEFAULT_RESULTS_DIR
        exporter = make_output_manager(self.mesh, output_dir)
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
            c_h, u_h = self._SolvePostprocessing(t_val)

            # Update old solutions
            self._UpdateOldState(u_h)

            # Save solution
            exporter.save(c_h, t_val)

        # Close output file
        exporter.close()

    def ConvergenceTest(self, t0, dt, T, nu, l, tol, maxIt, output_dir=None):

        self._ValidateInput(t0, dt, nu ,T, l)

        # Mesh data
        x     = SpatialCoordinate(self.mesh)
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0

        # Time loop parameters
        t_val   = t0 - (nu - 1)*dt 
        nsteps  = round((T - t_val)/dt)
        t       = Constant(t_val)
        self.nu = nu

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
        self.u_old[-1].assign(project(self.T.inv(self.c_ex), self.W))
        for i in range(1, nu):
            t_val += dt
            t.assign(t_val)
            u_tmp = project(self.T.inv(self.c_ex), self.W)
            self.u_old[-1-i].assign(u_tmp)
        self._SetInitialCondition(self.c_ex)

        # Forms ansd solver
        self._BuildVariationalForms(dt)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize error lists
        E_c    = None
        E_grad = None

        # Time loop
        self.decimals = get_decimals(dt)
        for i in range(nsteps-nu+1):

            # Update time step
            t_val += dt
            t.assign(t_val)

            # Solve the problem
            self.solver.solve()

            # Print convergence iterations
            E_c, E_grad, u_h = self._ConvergenceTestPostprocessing(t_val)

            # Update old solutions
            self._UpdateOldState(u_h)

        return E_c, E_grad, h_avg