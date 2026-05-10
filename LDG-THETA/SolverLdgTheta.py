from dolfin import *
import ufl
from pathlib import Path

from Utilities.EnumUtilities import SpaceMethod, TimeMethod

set_log_active(False)
parameters["ghost_mode"] = "shared_facet"

class SolverLdgTheta:
    """!
    Class for solving Fisher-Kolmogoro equation.
    
    - **Space discretization:** LDG
    - **Time discretization:** Theta-method
    """
    # Private: ______________________________________________________________________________ 
    # Constructor
    def __init__(self, mesh, D, alpha, C11, C12, c_0):
        """!
        Initializes the solver with physical parameters and mesh data.

        @param mesh  The dolfin.Mesh object representing the computational domain.
        @param D     Diffusion tensor (can be a Constant, Expression, or Matrix).
        @param alpha Linear reaction coefficient for the Fisher-Kolmogorov model.
        @param C11   Penalty parameter for concentration jumps (LDG stability).
        @param C12   Flux weight parameter (usually set as 0.5 * n('+')).
        @param c_0   Initial condition or analytical solution for convergence tests(lambda function of x and t).
        """
        self.mesh  = mesh
        self.D     = as_tensor(D)
        self.alpha = alpha if isinstance(alpha, ufl.core.expr.Expr) else Constant(alpha)
        self.C11   = C11   if isinstance(C11,   ufl.core.expr.Expr) else Constant(C11)
        self.C12   = C12   if isinstance(C12,   ufl.core.expr.Expr) else Constant(C12)
        self.c_0   = c_0  # This will be used as exact solution if run with ConvergenceTest
        self.SM    = SpaceMethod.LDG
        self.TM    = TimeMethod.THETA

    # Functional spaces constructor
    def _BuildFunctionSpaces(self, l=1):
        """!
        Constructs the mixed function space for the LDG formulation.

        @param l  Polynomial degree for the DG spaces (default: 1).
        """
        self.W        = FunctionSpace(self.mesh, "DG", l)
        self.R        = VectorFunctionSpace(self.mesh, "DG", l)
        element_c     = self.W.ufl_element()
        element_q     = self.R.ufl_element()
        mixed_element = MixedElement([element_c, element_q])
        self.WR       = FunctionSpace(self.mesh, mixed_element)

    # Functions constructor
    def _BuildFunctions(self):
        """!
        Initializes the functions and test functions on the mixed space.

        The method defines the solution function U, the old solution U_old 
        for the time stepping, and the test functions Phi.
        """
        self.U         = Function(self.WR)
        self.U_old     = Function(self.WR)

    # Variational forms builder(homogeneous Neumann BCs for the moment)
    def _BuildVariationalForms(self, tau, tht):
        """!
        Constructs the LDG variational forms for the coupled system.

        @param tau        Time step size (Constant).
        @param tht        Theta-method parameter (Constant: 0 explicit, 0.5 CN, 1 implicit).
        """
        # Geometry
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0
        n     = FacetNormal(self.mesh)

        # Data
        tau   = Constant(tau)
        tht   = Constant(tht)
        D     = self.D
        alpha = self.alpha
        C11   = self.C11
        C12   = self.C12*n('+')

        # Extract functions
        (c, q)         = split(self.U)
        (c_old, q_old) = split(self.U_old)
        Phi            = TestFunction(self.WR)
        (v, r)         = split(Phi)

        # Force term and Neumann BC
        Force     = self.Force
        Force_old = self.Force_old
        gN        = self.gN
        gN_old    = self.gN_old

        # Measures
        dx = Measure("dx", domain=self.mesh)
        dS = Measure("dS", domain=self.mesh)
        ds = Measure("ds", domain=self.mesh)

        # Forms
        Fq = inner(q, r)*dx \
            + inner(c, div(dot(D.T, r)))*dx \
            - ( avg(c) + inner(C12, jump(c, n)) )*jump(dot(D.T, r), n)*dS \
            - c*inner(dot(D.T, r), n)*ds
        Fc = (1.0/tau)*c*v*dx - (1.0/tau)*c_old*v*dx \
            + tht*inner(q, grad(v))*dx \
            + (1.0 - tht)*inner(q_old, grad(v))*dx \
            - tht*inner(avg(q) - (C11/h_avg)*jump(c, n) - C12*jump(q, n), jump(v, n))*dS \
            - (1.0 - tht)*inner(avg(q_old) - (C11/h_avg)*jump(c_old, n) - C12*jump(q_old, n), jump(v, n))*dS \
            - alpha*(tht*c + (1.0 - tht)*c_old)*(1.0 - (tht*c + (1.0 - tht)*c_old))*v*dx \
            - tht*Force*v*dx - (1.0 - tht)*Force_old*v*dx \
            - tht*inner(gN, n)*v*ds - (1.0 - tht)*inner(gN_old, n)*v*ds
        self.Form = Fq + Fc

    # Nonlinear solver builder
    def _BuildNonlinearSolver(self, tol=1e-8, maxIt=200):
        """!
        Sets up the SNES non-linear solver and the Jacobian matrix.

        @param tol    Absolute and relative tolerance for the Newton solver.
        @param maxIt  Maximum number of iterations for the non-linear solver.
        """
        # Jacobian
        dU = TrialFunction(self.WR)
        J  = derivative(self.Form, self.U, dU)

        # Nonlinear solver
        problem = NonlinearVariationalProblem(self.Form, self.U, J=J)
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

    # Method to update old steps
    def _UpdateOldState(self):
        """!
        Updates the state for the next time step.
        """
        self.U_old.assign(self.U)
        self.Force_old.assign(project(self.Force, self.W))
        self.gN_old.assign(project(self.gN, self.R))

    # Method to check correctness of the input parameters
    def _ValidateInput(self, t0, dt, T, tht, l):
        """! Internal validation of simulation parameters. """
        if not (0.0 <= tht <= 1.0):
            raise ValueError("tht must be between in [0, 1]")
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        if T <= t0:
            raise ValueError(f"Final time T ({T}) must be greater than initial time t0 ({t0})")
        if not isinstance(l, int) or l < 0:
            raise ValueError(f"Polynomial degree l must be a non-negative integer, got {l}")

    # Public: ________________________________________________________________________________
    # --- Method for solving FK equation ---
    def Solve(self, t0, dt, T, tht, l, tol, maxIt, extForce=None, NeumannBC=None):
        """!
        Executes the time-loop to solve the Fisher-Kolmogorov equation.

        @param t0     Initial time.
        @param dt     Time step size.
        @param T      Final simulation time.
        @param tht    Theta-method parameter.
        @param l      Polynomial degree for DG spaces.
        @param tol    Solver tolerance.
        @param maxIt  Maximum solver iterations.
        @param extForce  External forcing term (lambda function of x and t o Constant, optional).
        @param NeumannBC Neumann boundary condition (lambda function of x and t o Constant, optional).
        """
        self._ValidateInput(t0, dt, T, tht, l)
        
        # Mesh data
        x     = SpatialCoordinate(self.mesh)
        N_el  = self.mesh.num_cells()
        h_min = self.mesh.hmin()
        h_max = self.mesh.hmax()
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0

        # Time loop parameters
        t_val  = t0 
        nsteps = round((T - t0)/dt)
        t      = Constant(t0)

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Force and Neumann BC
        self.Force     = extForce(x, t) if extForce else Constant(0.0)
        self.gN        = NeumannBC(x, t) if NeumannBC else Constant((0.0, 0.0))
        self.Force_old = Function(self.W)
        self.gN_old    = Function(self.R)

        # Initial guess for solver and old terms
        assign(self.U.sub(0), project(self.c_0(x, t), self.W))
        assign(self.U.sub(1), project(dot(self.D, grad(self.c_0(x, t))), self.R))
        self._UpdateOldState() # This

        # Variational form and solver
        self._BuildVariationalForms(dt, tht)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize file
        output_path = Path(__file__).parent / "results" / self.mesh.name()
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = XDMFFile(self.mesh.mpi_comm(), str(output_path / "concentration.xdmf"))
        output_file.parameters["flush_output"] = True
        output_file.parameters["rewrite_function_mesh"] = False
        output_file.parameters["functions_share_mesh"] = True

        # Save initial condition
        c_h = self.U.sub(0)
        c_h.rename("concentration", "c")
        output_file.write(c_h, t_val)

        # Time loop
        for i in range(nsteps):
            # Update time step
            t_val += dt
            t.assign(t_val)

            # Solve the problem
            self.solver.solve()

            # Extract solution
            (c_h, q_h) = self.U.split(deepcopy=True)

            # Compute min and max values
            c_min = c_h.vector().min()
            c_max = c_h.vector().max()
            q_min = q_h.vector().min()
            q_max = q_h.vector().max()

            # Update old solutions
            self._UpdateOldState()

            # Print the bounds for both the variables
            print(f"N_el={N_el:1d}, l={l:1d}, t={t_val:.6f}, tht={tht:.1f}")
            print(f"  h   ∈ [{h_min:.6f}, {h_max:.6f}],  h_avg = {h_avg:.6f}")
            print(f"  c_h ∈ [{c_min:7.6f}, {c_max:7.6f}]")
            print(f"  q_h ∈ [{q_min:7.6f}, {q_max:7.6f}]")
            print("-"*70)

            # Save solution
            output_file.write(c_h, t_val)

        # Close output file
        output_file.close()

    # --- Method for a convergence test using c_0 as exact solution ---
    def ConvergenceTest(self, t0, dt, T, tht, l, tol, maxIt):
        """!
        Performs a convergence analysis using c_0 as the exact solution.

        @param t0     Initial time.
        @param dt     Time step size.
        @param T      Final simulation time.
        @param tht    Theta-method parameter.
        @param l      Polynomial degree for DG spaces.
        @param tol    Solver tolerance.
        @param maxIt  Maximum solver iterations.

        @return E_c   L2 error of the concentration.
        @return E_q   L2 error of the auxiliary flux.
        @return h_avg Average mesh size.
        """
        self._ValidateInput(t0, dt, T, tht, l)

        # Mesh data
        x     = SpatialCoordinate(self.mesh)
        N_el  = self.mesh.num_cells()
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0
        dx    = Measure("dx", domain=self.mesh)

        # Time loop parameters
        t_val  = t0 
        nsteps = round((T - t0)/dt)
        t      = Constant(t0)

        # Data
        D     = self.D
        alpha = self.alpha
        c_ex  = self.c_0(x, t)

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Computing forcing term
        c_t        = diff(c_ex, t)
        Delta_c    = div(D*grad(c_ex))
        self.Force = c_t - Delta_c - alpha*c_ex*(1.0 - c_ex)

        # Computing exact gradient and Neumann BC
        q_ex = dot(D, grad(c_ex))
        self.gN   = q_ex

        # Initial guess for solver and old terms
        self.Force_old = Function(self.W)
        self.gN_old    = Function(self.R)
        assign(self.U.sub(0), project(c_ex, self.W))
        assign(self.U.sub(1), project(q_ex, self.R))
        self._UpdateOldState()

        # Forms ansd solver
        self._BuildVariationalForms(dt, tht)
        self._BuildNonlinearSolver(tol, maxIt)

        # Initialize error lists
        E_c = None
        E_q = None

        # Time loop
        for i in range(nsteps):
            # Update time step
            t_val += dt
            t.assign(t_val)

            # Solve the problem
            self.solver.solve()

            # Extract solution
            (c_h, q_h) = self.U.split(deepcopy=True)

            # Compute errors
            E_c = sqrt(assemble((c_ex - c_h)*(c_ex - c_h)*dx))
            E_q = sqrt(assemble(inner(q_ex - q_h, q_ex - q_h )*dx))

            # Compute min and max values
            c_min = c_h.vector().min()
            c_max = c_h.vector().max()
            q_min = q_h.vector().min()
            q_max = q_h.vector().max()

            # Update old solutions
            self._UpdateOldState()

            # Print the bounds for both the variables
            print(f"N_el={N_el:1d}, h={h_avg:.6f}, l={l:1d}, t={t_val:.6f}, tht={tht:.1f}")
            print(f"  ||c_ex - c_h||_L2 = {E_c:10.6e}")
            print(f"  c_h ∈ [{c_min:7.6f}, {c_max:7.6f}]")
            print(f"  ||D*grad(c_ex) - q_h||_L2 = {E_q:10.6e}")
            print(f"  q_h ∈ [{q_min:7.6f}, {q_max:7.6f}]")
            print("-"*70)

        return E_c, E_q, h_avg