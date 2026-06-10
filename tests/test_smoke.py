"""
Smoke tests — one per solver × time method.

Each test builds a minimal mesh, instantiates the solver, and runs a single
time step. No accuracy is checked; the only assertion is that the run
completes without raising an exception.
"""

import pytest
from dolfin import *

from fisher_kolmogorov.models.solver_factory    import make_solver_class
from fisher_kolmogorov.configs.model_configs    import DgParams, LdgParams, PpDgParams, SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities import (
    PolyDegree, BdfOrder, ThetaMethod, SpaceMethod, TimeMethod,
)


# Shared fixtures _________________________________________________________________________________________________________________________
@pytest.fixture(scope="module")
def minimal_mesh():
    return UnitSquareMesh(4, 4)


@pytest.fixture(scope="module")
def pde_data():
    D     = Identity(2)
    alpha = Constant(1.0)

    def c_0(x, t):
        return 0.25 * (cos(2 * pi * x[0]) * cos(2 * pi * x[1]) + 2.0) * (1.0 - t)

    return D, alpha, c_0


# Helpers __________________________________________________________________________________________________________________________________
def _run_one_step_bdf(solver_class, mesh, D, alpha, c_0):
    solver = solver_class(mesh, D, alpha, c_0)
    solver.ConvergenceTest(
        t0=0.0, dt=1e-2, T=1e-2,
        time_order=BdfOrder.BDF1, l=PolyDegree.P1,
        tol=1e-6, maxIt=50,
    )


def _run_one_step_theta(solver_class, mesh, D, alpha, c_0):
    solver = solver_class(mesh, D, alpha, c_0)
    solver.ConvergenceTest(
        t0=0.0, dt=1e-2, T=1e-2,
        time_order=ThetaMethod.CN, l=PolyDegree.P1,
        tol=1e-6, maxIt=50,
    )


# Tests smoke on DG methods ______________________________________________________________________________________________________________
def test_dg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.DG, TimeMethod.BDF, DgParams())
    _run_one_step_bdf(solver_class, minimal_mesh, D, alpha, c_0)


def test_dg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.DG, TimeMethod.THETA, DgParams())
    _run_one_step_theta(solver_class, minimal_mesh, D, alpha, c_0)


# Tests smoke on LDG methods ______________________________________________________________________________________________________________
def test_ldg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.LDG, TimeMethod.BDF, LdgParams())
    _run_one_step_bdf(solver_class, minimal_mesh, D, alpha, c_0)


def test_ldg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.LDG, TimeMethod.THETA, LdgParams())
    _run_one_step_theta(solver_class, minimal_mesh, D, alpha, c_0)


# Tests smoke on PP-DG methods ____________________________________________________________________________________________________________
def test_ppdg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.PPDG, TimeMethod.BDF, PpDgParams())
    _run_one_step_bdf(solver_class, minimal_mesh, D, alpha, c_0)


def test_ppdg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.PPDG, TimeMethod.THETA, PpDgParams())
    _run_one_step_theta(solver_class, minimal_mesh, D, alpha, c_0)


# Tests smoke on SP-LDG methods ___________________________________________________________________________________________________________
def test_spldg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.SPLDG, TimeMethod.BDF, SpLdgParams())
    _run_one_step_bdf(solver_class, minimal_mesh, D, alpha, c_0)


def test_spldg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.SPLDG, TimeMethod.THETA, SpLdgParams())
    _run_one_step_theta(solver_class, minimal_mesh, D, alpha, c_0)


def test_spldg_bdf_red2_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    solver_class = make_solver_class(SpaceMethod.SPLDG, TimeMethod.BDF, SpLdgParams(), full=False)
    _run_one_step_bdf(solver_class, minimal_mesh, D, alpha, c_0)