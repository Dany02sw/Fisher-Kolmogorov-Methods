from dolfin import *
from config import DEFAULT_RESULTS_DIR

from fisher_kolmogorov.utilities.fenics_utilities import Normalize
from fisher_kolmogorov.utilities.enum_utilities   import TimeMethod
from fisher_kolmogorov.utilities.io_utilities     import make_output_manager
from fisher_kolmogorov.utilities.math_utilities   import get_decimals

_BUTCHER = {
    2: {
        # Heun's method (explicit trapezoidal)
        "c": [0.0, 1.0],
        "A": [[0.0, 0.0],
              [1.0, 0.0]],
        "b": [0.5, 0.5],
    },
    4: {
        # Classical RK4
        "c": [0.0, 0.5, 0.5, 1.0],
        "A": [[0.0, 0.0, 0.0, 0.0],
              [0.5, 0.0, 0.0, 0.0],
              [0.0, 0.5, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]],
        "b": [1/6, 1/3, 1/3, 1/6],
    },
    6: {
        # Butcher (1964), 7-stage order-6 explicit method.
        # Ref: Butcher, J.C., J. Austral. Math. Soc. 4 (1964), 179-194.
        "c": [0.0, 1/3, 2/3, 1/3, 5/6, 1/6, 1.0],
        "A": [
            [ 0.0,       0.0,      0.0,      0.0,      0.0,     0.0,  0.0],
            [ 1/3,       0.0,      0.0,      0.0,      0.0,     0.0,  0.0],
            [ 0.0,       2/3,      0.0,      0.0,      0.0,     0.0,  0.0],
            [ 1/12,      1/3,     -1/12,     0.0,      0.0,     0.0,  0.0],
            [ 25/48,    -55/24,    35/48,    15/8,     0.0,     0.0,  0.0],
            [ 3/20,     -11/24,   -1/8,      1/2,      1/10,    0.0,  0.0],
            [-261/260,   33/13,    43/156,  -118/39,   32/195,  80/39, 0.0],
        ],
        "b": [13/200, 0.0, 11/40, 11/40, 4/25, 4/25, 13/200],
    },
}

# Maps BDF order to the minimum RK order needed to match its accuracy
_BDF_TO_RK = {
    1: 2,
    2: 2,
    3: 4,
    4: 4,
    5: 6,
    6: 6
}


class TimeMixinRk:
    """Explicit Runge-Kutta time discretization mixin (orders 2, 4, 6).

    Serves two purposes:
      1. Stand-alone time integration via Solve / ConvergenceTest,
         following the same interface as TimeMixinBdf and TimeMixinTheta.
      2. BDF startup: BuildStartupChain produces the u_old list
         required by TimeMixinBdf when nu > 1, using sub-steps to
         satisfy the parabolic CFL condition.

    RK order selection (used both stand-alone and for startup):
        BDF 1-2  ->  RK2  (Heun)
        BDF 3-4  ->  RK4  (classical)
        BDF 5-6  ->  RK6  (Butcher 1964, 7-stage)

    Note: does not call _BuildNonlinearSolver — time stepping is explicit.
    """

    def _init_time(self):
        self.TM        = TimeMethod.RK
        self.W         = None
        self._rk_order = 4  # default; overridden by time_order in Solve/ConvergenceTest

    def _BuildFunctions(self):
        self.U = Function(self.WR)

    def _BuildTimeForm(self, tau, u, v):
        # RK does not use a UFL time form; time integration is handled
        # explicitly in _RKStep. Intentionally a no-op.
        return None

    def _BuildVariationalForms(self, tau):
        """Build only the spatial residual; time stepping is explicit."""
        self.tau = Constant(tau)

        F_space   = self._BuildSpatialForm()
        comps     = split(self.U) if self.WR != self.W else (self.U,)
        transf    = self.T(comps[0])
        F, _, _   = F_space(comps, comps, transf, self.Force, self.gN)
        self.Form = F

    def _UpdateOldState(self, u):
        # No old-state chain for a one-step explicit method
        pass

    def _ValidateInput(self, t0, dt, rk_order, T, l):
        super()._ValidateInput(t0, dt, T, l)
        if rk_order not in _BUTCHER:
            raise ValueError(f"rk_order must be one of {list(_BUTCHER.keys())}, got {rk_order}")


    def _RKStep(self, U_n: Function, dt: float) -> Function:
        """Advance U_n by one explicit RK step of size dt.

        The spatial residual is re-evaluated at each stage by temporarily
        assigning the stage value to self.U and re-assembling the form.
        """
        tableau = _BUTCHER[self._rk_order]
        A, b, c = tableau["A"], tableau["b"], tableau["c"]
        s       = len(b)

        K = []

        for i in range(s):
            # Stage value: U_stage = U_n + dt * sum_{j<i} A[i][j] * K[j]
            U_stage = Function(self.WR)
            U_stage.assign(U_n)
            for j in range(i):
                if abs(A[i][j]) > 1e-15:
                    U_stage.vector().axpy(dt * A[i][j], K[j].vector())

            # Evaluate the spatial residual at U_stage
            self.U.assign(U_stage)
            rhs = assemble(self.Form)

            # Store the stage increment K[i] = -M^{-1} F(U_stage)
            K_i = Function(self.WR)
            K_i.vector()[:] = rhs[:]
            K.append(K_i)

        # Update: U_{n+1} = U_n + dt * sum_i b[i] * K[i]
        U_next = Function(self.WR)
        U_next.assign(U_n)
        for i in range(s):
            if abs(b[i]) > 1e-15:
                U_next.vector().axpy(dt * b[i], K[i].vector())

        return U_next


    def BuildStartupChain(self, t0: float, dt_bdf: float, nu: int,
                          l: int, tol: float, maxIt: int,
                          extForce=None, NeumannBC=None,
                          substeps: int = 10) -> list:
        """Build the u_old chain required by TimeMixinBdf when nu > 1.

        Advances the initial condition by (nu - 1) BDF steps using an
        explicit RK method whose order matches the BDF order (via
        _BDF_TO_RK). Each BDF step dt_bdf is subdivided into `substeps`
        explicit sub-steps to satisfy the parabolic CFL condition.

        Parameters
        ----------
        t0        : float  –  initial time
        dt_bdf    : float  –  BDF time step (used to space the chain entries)
        nu        : int    –  BDF order (1..6)
        l         : int    –  polynomial degree
        tol       : float  –  kept for interface consistency (unused)
        maxIt     : int    –  kept for interface consistency (unused)
        extForce  : callable or None
        NeumannBC : callable or None
        substeps  : int    –  number of explicit sub-steps per BDF step

        Returns
        -------
        list of Function on self.W, length nu, ordered as u_old:
        index 0 = most recent (t0), index nu-1 = oldest (t0 - (nu-1)*dt_bdf).
        """
        if nu not in _BDF_TO_RK:
            raise ValueError(f"nu must be in 1..6, got {nu}")

        if nu == 1:
            # BDF1 needs only the current state; no startup required
            x   = SpatialCoordinate(self.mesh)
            t   = Constant(t0)
            self._BuildFunctionSpaces(l)
            c_0 = project(self.c_0(x, t), self.W)
            c_0 = Normalize(c_0)
            return [project(self.T.inv(c_0), self.W)]

        self._rk_order = _BDF_TO_RK[nu]

        x      = SpatialCoordinate(self.mesh)
        t      = Constant(t0)
        dt_sub = dt_bdf / substeps

        self._BuildFunctionSpaces(l)
        self._BuildFunctions()
        self._SetSourceTerm(x, t, extForce, NeumannBC)

        c_0 = project(self.c_0(x, t), self.W)
        c_0 = Normalize(c_0)
        self._SetInitialCondition(c_0)
        self._BuildVariationalForms(dt_sub)

        # snapshots[0] = t0 (most recent), filled in reverse at the end
        snapshots = [Function(self.W)]
        snapshots[0].assign(project(self.T.inv(c_0), self.W))

        U_prev = Function(self.WR)
        U_prev.assign(self.U)

        t_val = t0
        for k in range(nu - 1):
            # Advance by one BDF step using substeps explicit sub-steps
            for _ in range(substeps):
                t_val += dt_sub
                t.assign(t_val)
                U_next = self._RKStep(U_prev, dt_sub)
                U_prev.assign(U_next)

            # Extract scalar component and store
            snap = Function(self.W)
            if self.WR != self.W:
                snap.assign(project(split(U_prev)[0], self.W))
            else:
                snap.assign(U_prev)
            snapshots.append(snap)

        # Reverse so index 0 = most recent, index nu-1 = oldest,
        # matching the layout of TimeMixinBdf.u_old
        snapshots.reverse()
        return snapshots
    

    def Solve(self, t0, dt, T, time_order, l, tol, maxIt,
              extForce=None, NeumannBC=None, output_dir=None):

        self._ValidateInput(t0, dt, time_order, T, l)
        self._rk_order = time_order

        x      = SpatialCoordinate(self.mesh)
        t      = Constant(t0)
        t_val  = t0
        nsteps = round((T - t0) / dt)

        self._BuildFunctionSpaces(l)
        self._BuildFunctions()
        self._SetSourceTerm(x, t, extForce, NeumannBC)

        c_0 = project(self.c_0(x, t), self.W)
        c_0 = Normalize(c_0)
        self._SetInitialCondition(c_0)
        self._BuildVariationalForms(dt)

        if output_dir is None:
            output_dir = DEFAULT_RESULTS_DIR
        exporter = make_output_manager(self.mesh, output_dir)
        exporter.open()
        exporter.save(c_0, t_val)

        self.decimals = get_decimals(dt)
        U_prev = Function(self.WR)
        U_prev.assign(self.U)

        for i in range(nsteps):
            # Update time step
            t_val += dt
            t.assign(t_val)

            U_next = self._RKStep(U_prev, dt)
            self.U.assign(U_next)
            U_prev.assign(U_next)

            c_h, _ = self._SolvePostprocessing(t_val)
            exporter.save(c_h, t_val)

        exporter.close()


    def ConvergenceTest(self, t0, dt, T, time_order, l, tol, maxIt,
                        output_dir=None):

        self._ValidateInput(t0, dt, time_order, T, l)
        self._rk_order = time_order

        x      = SpatialCoordinate(self.mesh)
        h_avg  = (self.mesh.hmax() + self.mesh.hmin()) / 2.0
        t_val  = t0
        nsteps = round((T - t0) / dt)
        t      = Constant(t_val)

        D         = self.D
        alpha     = self.alpha
        self.c_ex = self.c_0(x, t)

        self._BuildFunctionSpaces(l)
        self._BuildFunctions()

        # Computing forcing term
        c_t        = diff(self.c_ex, t)
        Delta_c    = div(D * grad(self.c_ex))
        self.Force = c_t - Delta_c - alpha * self.c_ex * (1.0 - self.c_ex)

        # Computing Neumann BC
        self.gN = dot(D, grad(self.c_ex))

        self._SetInitialCondition(self.c_ex)
        self._BuildVariationalForms(dt)

        E_c    = None
        E_grad = None

        self.decimals = get_decimals(dt)
        U_prev = Function(self.WR)
        U_prev.assign(self.U)

        for i in range(nsteps):
            # Update time step
            t_val += dt
            t.assign(t_val)

            U_next = self._RKStep(U_prev, dt)
            self.U.assign(U_next)
            U_prev.assign(U_next)

            # Print convergence iterations
            E_c, E_grad, _, _ = self._ConvergenceTestPostprocessing(t_val)

        return E_c, E_grad, h_avg
