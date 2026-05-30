#####################################################
# DA RIVEDERE + AGGIUNGERE IMPLICIT RK ##############
#####################################################

"""
rk_start.py
===========
Runge-Kutta startup chain for BDF multistep methods in FEniCS.

RKStart operates on the full mixed space WR so that the spatial form
is evaluated correctly for any formulation (LDG, HDG, 4-field, ...).
Only the component needed by BDF (typically the evolved scalar) is
extracted and returned, so u_old stays on the cheap scalar space W.

RK order selection
------------------
BDF 1-2  ->  RK2 (Heun)
BDF 3-4  ->  RK4 (classical)
BDF 5-6  ->  RK6 (Butcher 1964, 7-stage)
"""

from __future__ import annotations

import fenics as fe
from typing import Callable, List, Optional


_BUTCHER: dict[int, dict] = {
    2: {
        "c": [0.0, 1.0],
        "A": [[0.0, 0.0],
              [1.0, 0.0]],
        "b": [0.5, 0.5],
    },
    4: {
        "c": [0.0, 0.5, 0.5, 1.0],
        "A": [[0.0, 0.0, 0.0, 0.0],
              [0.5, 0.0, 0.0, 0.0],
              [0.0, 0.5, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]],
        "b": [1/6, 1/3, 1/3, 1/6],
    },
    # Butcher (1964) six-stage, order-6 explicit method.
    # Ref: Butcher, J.C., J. Austral. Math. Soc. 4 (1964), 179-194.
    6: {
        "c": [0.0, 1/3, 2/3, 1/3, 5/6, 1/6, 1.0],
        "A": [
            [0.0,        0.0,      0.0,      0.0,       0.0,      0.0,  0.0],
            [1/3,        0.0,      0.0,      0.0,       0.0,      0.0,  0.0],
            [0.0,        2/3,      0.0,      0.0,       0.0,      0.0,  0.0],
            [1/12,       1/3,     -1/12,     0.0,       0.0,      0.0,  0.0],
            [25/48,     -55/24,    35/48,    15/8,      0.0,      0.0,  0.0],
            [3/20,      -11/24,   -1/8,      1/2,       1/10,     0.0,  0.0],
            [-261/260,   33/13,    43/156,  -118/39,    32/195,   80/39, 0.0],
        ],
        "b": [13/200, 0.0, 11/40, 11/40, 4/25, 4/25, 13/200],
    },
}

_BDF_TO_RK: dict[int, int] = {1: 2, 2: 2, 3: 4, 4: 4, 5: 6, 6: 6}


class RKStart:
    """
    Builds the old-solution chain required by BDF-p.

    Stage solves are performed on WR (full mixed space) so that the
    spatial form is consistent with the actual discretisation.
    The returned chain lives on the space determined by extract_fn,
    which defaults to the identity (returns a Function on WR).

    Parameters
    ----------
    WR           : FunctionSpace  –  mixed space (SolverBase.WR)
    U0           : Function       –  initial condition on WR
    form_fn      : callable  (U_new, U_old, dt) -> ufl.Form
                   Full residual on WR; test function is internal.
    nu           : int  –  BDF order (1..6)
    dt           : float
    bcs          : list of DirichletBC on WR
    extract_fn   : callable  Function(WR) -> Function(V_cheap)
                   Applied to each solution before storing in the chain.
                   Use  lambda U: U.sub(0)  to keep only the scalar
                   component and avoid storing the full mixed vector.
                   Defaults to identity (stores full WR Functions).
    solver_params : dict  –  forwarded to NonlinearVariationalSolver
    """

    def __init__(
        self,
        WR            : fe.FunctionSpace,
        U0            : fe.Function,
        form_fn       : Callable,
        nu            : int,
        dt            : float,
        bcs           : Optional[List]     = None,
        extract_fn    : Optional[Callable] = None,
        solver_params : Optional[dict]     = None,
    ):
        if nu not in _BDF_TO_RK:
            raise ValueError(f"nu must be in 1..6, got {nu}.")

        self.WR            = WR
        self.dt            = dt
        self.nu            = nu
        self.bcs           = bcs or []
        self.extract_fn    = extract_fn if extract_fn is not None else (lambda U: U)
        self.solver_params = solver_params or {}
        self.tableau       = _BUTCHER[_BDF_TO_RK[nu]]
        self.form_fn       = form_fn

        self._U0 = fe.Function(WR, name="U_0")
        self._U0.assign(U0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[fe.Function]:
        """
        Advance U0 by (nu - 1) RK steps of size dt.

        Returns [U_{n-1}, ..., U_0] with extract_fn applied to each
        entry, matching the layout of SolverBDF.u_old (index 0 = most
        recent, index -1 = oldest).
        """
        if self.nu == 1:
            return [self.extract_fn(self._U0)]

        chain: List[fe.Function] = []
        U_prev = fe.Function(self.WR)
        U_prev.assign(self._U0)

        for k in range(self.nu - 1):
            U_next = self._rk_step(U_prev)
            chain.append(U_next)
            U_prev = U_next

        # chain = [U_1, ..., U_{nu-1}]; apply extract_fn and reverse
        # so that index 0 is the most recent (U_{nu-1}) and the
        # appended U_0 is the oldest, matching u_old layout.
        chain = [self.extract_fn(U) for U in chain]
        chain.reverse()
        chain.append(self.extract_fn(self._U0))
        return chain

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rk_step(self, U_n: fe.Function) -> fe.Function:
        """
        Single RK step U_n -> U_{n+1} on WR.

        extract_fn is intentionally NOT applied here; extraction happens
        once in run() after all stages are complete, so intermediate
        stage values remain on WR throughout.
        """
        A, b   = self.tableau["A"], self.tableau["b"]
        s      = len(b)
        dt     = self.dt
        dt_one = fe.Constant(1.0)

        K: List[fe.Function] = []

        for i in range(s):
            U_stage = fe.Function(self.WR)
            U_stage.assign(U_n)
            for j in range(i):
                if abs(A[i][j]) > 1e-15:
                    U_stage.vector().axpy(dt * A[i][j], K[j].vector())
            for bc in self.bcs:
                bc.apply(U_stage.vector())

            U_new = fe.Function(self.WR)
            U_new.assign(U_stage)
            F       = self.form_fn(U_new, U_stage, dt_one)
            J       = fe.derivative(F, U_new)
            problem = fe.NonlinearVariationalProblem(F, U_new, bcs=self.bcs, J=J)
            solver  = fe.NonlinearVariationalSolver(problem)
            if self.solver_params:
                solver.parameters.update(self.solver_params)
            solver.solve()

            K_i = fe.Function(self.WR)
            K_i.vector()[:] = U_new.vector() - U_stage.vector()
            K.append(K_i)

        U_next = fe.Function(self.WR)
        U_next.assign(U_n)
        for i in range(s):
            if abs(b[i]) > 1e-15:
                U_next.vector().axpy(dt * b[i], K[i].vector())
        for bc in self.bcs:
            bc.apply(U_next.vector())

        return U_next