from dolfin import *
import os

set_log_active(False)
parameters["ghost_mode"] = "shared_facet"

class SolverLdgTheta:
    # Private: ______________________________________________________________________________ 
    # Constructor
    def __init__(self, mesh, D, alpha, C11, C12, c_0):
        self.mesh  = mesh
        self.D     = D
        self.alpha = alpha
        self.C11   = C11
        self.C12   = C12
        self.c_0   = c_0  # This will be used as exact solution if run with ConvergenceTest

    # Functional spaces constructor
    def _BuildFunctionSpaces(self, l = 1):
        self.W        = FunctionSpace(self.mesh, "DG", l)
        self.R        = VectorFunctionSpace(self.mesh, "DG", l)
        element_c     = self.W.ufl_element()
        element_q     = self.R.ufl_element()
        mixed_element = MixedElement([element_c, element_q])
        self.WR       = FunctionSpace(self.mesh, mixed_element)

    # Functions constructor
    def _BuildFunctions(self):
        self.U     = Function(self.WR)
        self.U_old = Function(self.WR)
        self.Phi   = TestFunction(self.WR)

    # Variational forms builder(homogeneous Neumann BCs for the moment)
    def _BuildVariationalForms(self, tau, tht, Force, Force_old, gN, gN_old):
        # Geometry
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0
        n     = FacetNormal(self.mesh)

        # Data
        tau   = Constant(tau)
        D     = as_tensor(self.D)
        alpha = Constant(self.alpha)
        tht   = Constant(tht)
        C11   = Constant(self.C11)
        C12   = Constant(self.C12)*n('+')

        # Extract functions
        (c, q)         = split(self.U)
        (c_old, q_old) = split(self.U_old)
        (v, r)         = split(self.Phi)

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
        # Jacobian
        self.dU = TrialFunction(self.WR)
        self.J  = derivative(self.Form, self.U, self.dU)

        # Nonlinear solver
        self.problem = NonlinearVariationalProblem(self.Form, self.U, J=self.J)
        self.solver  = NonlinearVariationalSolver(self.problem)
        prm          = self.solver.parameters

        prm['nonlinear_solver']                  = 'snes'
        prm['snes_solver']['method']             = 'newtonls'
        prm['snes_solver']['line_search']        = 'bt'
        prm['snes_solver']['absolute_tolerance'] = tol
        prm['snes_solver']['relative_tolerance'] = tol
        prm['snes_solver']['maximum_iterations'] = maxIt
        prm['snes_solver']['linear_solver']      = 'lu'
        prm['snes_solver']['preconditioner']     = 'none'

    # Public: ________________________________________________________________________________
    # --- Method for solving FK equation ---
    def Solve(self, t0, dt, T, tht, l, tol, maxIt):
        # Mesh data
        x     = SpatialCoordinate(self.mesh)
        N_el  = self.mesh.num_cells()
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0

        # Data
        Force     = Constant(0.0)
        Force_old = Constant(0.0) 
        gN        = Constant((0.0, 0.0))
        gN_old    = Constant((0.0, 0.0))

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()
        self._BuildVariationalForms(dt, tht, Force, Force_old, gN, gN_old)
        self._BuildNonlinearSolver(tol, maxIt)

        # Time loop parameters
        t_val  = t0 
        t      = Constant(t0)
        nsteps = int((T - t0)/dt)

        # Initial guess for Nonlinear solver
        assign(self.U_old.sub(0), project(self.c_0(x, t), self.W))
        assign(self.U_old.sub(1), project(dot(self.D, grad(self.c_0(x, t))), self.R))
        self.U.assign(self.U_old)

        # Time loop
        for i in range(nsteps):
            # Update time step
            t_val += dt
            self.t.assign(t_val)

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
            self.U_old.assign(self.U)

            # Print the bounds for both the variables
            print(f"N_el={N_el:1d}, h={h_avg:.6f}, l={l:1d}, t={t_val:.6f}, tht={tht:.1f}")
            print(f"  c_h ∈ [{c_min:7.6f}, {c_max:7.6f}]")
            print(f"  q_h ∈ [{q_min:7.6f}, {q_max:7.6f}]")
            print("-"*70)

    # --- Method for a convergence test using c_0 as exact solution ---
    def ConvergenceTest(self, t0, dt, T, tht, l, tol, maxIt):
        # Mesh data
        x     = SpatialCoordinate(self.mesh)
        N_el  = self.mesh.num_cells()
        h_avg = (self.mesh.hmax() + self.mesh.hmin()) / 2.0
        dx    = Measure("dx", domain=self.mesh)

        # Data
        D     = as_tensor(self.D)
        alpha = Constant(self.alpha)
        t     = Constant(t0)
        c_ex  = self.c_0(x, t)

        # Computing forcing term
        c_t     = diff(c_ex, t)
        Delta_c = div(D*grad(c_ex))
        Force   = c_t - Delta_c - alpha*c_ex*(1.0 - c_ex)

        # Computing exact gradient and Neumann BC
        q_ex    = dot(D, grad(c_ex))
        gN      = q_ex

        # Functional setting 
        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Old terms
        Force_old = Function(self.W)
        gN_old    = Function(self.R)
        Force_old.assign(project(Force, self.W))
        gN_old.assign(project(q_ex, self.R))

        # Forms ansd solver
        self._BuildVariationalForms(dt, tht, Force, Force_old, gN, gN_old)
        self._BuildNonlinearSolver(tol, maxIt)

        # Time loop parameters
        t_val  = t0 
        nsteps = int((T - t0)/dt)

        # Initial guess for Nonlinear solver
        assign(self.U_old.sub(0), project(c_ex, self.W))
        assign(self.U_old.sub(1), project(dot(self.D, grad(c_ex)), self.R))
        self.U.assign(self.U_old)

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
            Force_old.assign(project(Force, self.W))
            gN_old.assign(project(q_ex, self.R))
            self.U_old.assign(self.U)

            # Print the bounds for both the variables
            print(f"N_el={N_el:1d}, h={h_avg:.6f}, l={l:1d}, t={t_val:.6f}, tht={tht:.1f}")
            print(f"  ||c_ex - c_h||_L2 = {E_c:10.6e}")
            print(f"  c_h ∈ [{c_min:7.6f}, {c_max:7.6f}]")
            print(f"  ||D*grad(c_ex) - q_h||_L2 = {E_q:10.6e}")
            print(f"  q_h ∈ [{q_min:7.6f}, {q_max:7.6f}]")
            print("-"*70)

        return E_c, E_q, h_avg

