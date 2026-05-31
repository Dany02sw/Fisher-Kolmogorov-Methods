"""
Smoke tests — one per solver × time method.

Each test builds a minimal mesh, instantiates the solver, and runs a single
time step. No accuracy is checked; the only assertion is that the run
completes without raising an exception.
"""

import pytest
from dolfin import *

from fisher_kolmogorov.models.solver_dg_bdf      import SolverDgBDF
from fisher_kolmogorov.models.solver_dg_theta    import SolverDgTheta
from fisher_kolmogorov.models.solver_ldg_bdf     import SolverLdgBDF
from fisher_kolmogorov.models.solver_ldg_theta   import SolverLdgTheta
from fisher_kolmogorov.models.solver_ppdg_bdf    import SolverPpDgBDF
from fisher_kolmogorov.models.solver_ppdg_theta  import SolverPpDgTheta
from fisher_kolmogorov.models.solver_spldg_bdf   import SolverSpLdgBDF
from fisher_kolmogorov.models.solver_spldg_theta import SolverSpLdgTheta
from fisher_kolmogorov.configs.model_configs     import DgParams, LdgParams, PpDgParams, SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities  import PolyDegree, BdfOrder, ThetaMethod


# ── Shared fixtures ────────────────────────────────────────────────────────────

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_one_step_bdf(solver_class, mesh, D, alpha, c_0, model_params):
    solver = solver_class(mesh, D, alpha, c_0, **model_params.to_kwargs())
    solver.ConvergenceTest(
        t0=0.0, dt=1e-2, T=1e-2,
        nu=BdfOrder.BDF1, l=PolyDegree.P1,
        tol=1e-6, maxIt=50,
    )


def _run_one_step_theta(solver_class, mesh, D, alpha, c_0, model_params):
    solver = solver_class(mesh, D, alpha, c_0, **model_params.to_kwargs())
    solver.ConvergenceTest(
        t0=0.0, dt=1e-2, T=1e-2,
        tht=ThetaMethod.CN, l=PolyDegree.P1,
        tol=1e-6, maxIt=50,
    )


# ── DG ─────────────────────────────────────────────────────────────────────────

def test_dg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_bdf(SolverDgBDF, minimal_mesh, D, alpha, c_0, DgParams())


def test_dg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_theta(SolverDgTheta, minimal_mesh, D, alpha, c_0, DgParams())


# ── LDG ────────────────────────────────────────────────────────────────────────

def test_ldg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_bdf(SolverLdgBDF, minimal_mesh, D, alpha, c_0, LdgParams())


def test_ldg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_theta(SolverLdgTheta, minimal_mesh, D, alpha, c_0, LdgParams())


# ── PP-DG ──────────────────────────────────────────────────────────────────────

def test_ppdg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_bdf(SolverPpDgBDF, minimal_mesh, D, alpha, c_0, PpDgParams())


def test_ppdg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_theta(SolverPpDgTheta, minimal_mesh, D, alpha, c_0, PpDgParams())


# ── SP-LDG ─────────────────────────────────────────────────────────────────────

def test_spldg_bdf_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_bdf(SolverSpLdgBDF, minimal_mesh, D, alpha, c_0, SpLdgParams())


def test_spldg_theta_smoke(minimal_mesh, pde_data):
    D, alpha, c_0 = pde_data
    _run_one_step_theta(SolverSpLdgTheta, minimal_mesh, D, alpha, c_0, SpLdgParams())