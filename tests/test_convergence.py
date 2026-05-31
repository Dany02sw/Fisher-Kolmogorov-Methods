"""
Convergence tests — one per spatial family.

Each test runs a spatial h-refinement study on the cosine problem and asserts
that the DG errors decrease monotonically across mesh refinements.
No convergence rate is checked; only monotone decrease is required.
"""

import pytest
from dolfin import *

from fisher_kolmogorov.models.solver_dg_bdf        import SolverDgBDF
from fisher_kolmogorov.models.solver_ldg_bdf       import SolverLdgBDF
from fisher_kolmogorov.models.solver_ppdg_bdf      import SolverPpDgBDF
from fisher_kolmogorov.models.solver_spldg_bdf     import SolverSpLdgBDF
from fisher_kolmogorov.configs.model_configs       import DgParams, LdgParams, PpDgParams, SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities    import PolyDegree, BdfOrder, MeshType, MeshStructure
from fisher_kolmogorov.configs.test_configs.cosine import COSINE_SPATIAL


# ── Helpers ────────────────────────────────────────────────────────────────────

def _convergence_params():
    """Return a lightweight override of COSINE_SPATIAL for fast testing."""
    from fisher_kolmogorov.configs.test_configs import ConvergenceParams
    from dataclasses import replace

    fast_spatial = ConvergenceParams(
        N_ref          = [2, 3, 4],
        l_space        = PolyDegree.P1,
        T              = 1e-2,
        dt             = 1e-2,
        nu_or_tht      = BdfOrder.BDF1,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.STRUCTURED,
    )
    return replace(COSINE_SPATIAL, spatial=fast_spatial)


def _assert_monotone_decrease(errors: list, label: str):
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f"{label}: error not decreasing at refinement {i+1} "
            f"({errors[i]:.3e} → {errors[i+1]:.3e})"
        )


def _collect_errors(solver_class, model_params):
    """Run spatial convergence and return the list of DG errors."""
    from fisher_kolmogorov.meshes.mesh_import import mesh_factory

    config  = _convergence_params()
    p       = config.spatial
    d_ext   = Constant(config.d_ext)
    D       = config.D_factory(d_ext, mesh=None)
    alpha   = config.alpha

    errors_dg = []
    for n in p.N_ref:
        N = 2 ** n
        mesh, _ = mesh_factory(
            mesh_type=p.mesh_type,
            N=N,
            structure=p.mesh_structure,
        )
        solver = solver_class(mesh, D, alpha, config.c_exact, **model_params.to_kwargs())
        _, E_DG, _ = solver.ConvergenceTest(
            t0=config.t0, dt=p.dt, T=p.T,
            nu=p.nu_or_tht, l=p.l_space,
            tol=1e-6, maxIt=50,
        )
        errors_dg.append(E_DG)

    return errors_dg


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_dg_spatial_convergence():
    errors = _collect_errors(SolverDgBDF, DgParams())
    _assert_monotone_decrease(errors, "DG-BDF")


def test_ldg_spatial_convergence():
    errors = _collect_errors(SolverLdgBDF, LdgParams())
    _assert_monotone_decrease(errors, "LDG-BDF")


def test_ppdg_spatial_convergence():
    errors = _collect_errors(SolverPpDgBDF, PpDgParams())
    _assert_monotone_decrease(errors, "PP-DG-BDF")


def test_spldg_spatial_convergence():
    errors = _collect_errors(SolverSpLdgBDF, SpLdgParams())
    _assert_monotone_decrease(errors, "SP-LDG-BDF")