"""
Convergence tests — one per spatial family.

Each test runs a spatial h-refinement study on the cosine problem and asserts
that the DG errors decrease monotonically across mesh refinements.
No convergence rate is checked; only monotone decrease is required.
"""

import pytest
from dolfin import *

from fisher_kolmogorov.models.solver_factory    import make_solver_class
from fisher_kolmogorov.configs.model_configs    import DgParams, LdgParams, PpDgParams, SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities import (
    PolyDegree, MeshStructure, SpaceMethod, TimeMethod,
)
from fisher_kolmogorov.configs.test_configs.cosine import make_cosine_spatial


# Helpers __________________________________________________________________________________________________________________________________
def _convergence_params():
    """Return a lightweight override of a spatial convergence test on cosine for fast testing."""
    return make_cosine_spatial(l_space=PolyDegree.P1, mesh_structure=MeshStructure.STRUCTURED)


def _assert_monotone_decrease(errors: list, label: str):
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f"{label}: error not decreasing at refinement {i+1} "
            f"({errors[i]:.3e} → {errors[i+1]:.3e})"
        )


def _collect_errors(solver_class):
    """Run spatial convergence and return the list of grad errors."""
    from fisher_kolmogorov.meshes.mesh_import import mesh_factory

    config = _convergence_params()
    p      = config.spatial
    d_ext  = Constant(config.d_ext)
    D      = config.D_factory(d_ext, mesh=None)
    alpha  = config.alpha

    errors_grad = []
    for n in p.N_ref:
        N = 2 ** n
        mesh, _ = mesh_factory(
            mesh_type=p.mesh_type,
            N=N,
            structure=p.mesh_structure,
        )
        solver = solver_class(mesh, D, alpha, config.c_exact)
        _, E_grad, _ = solver.ConvergenceTest(
            t0=config.t0, dt=p.dt, T=p.T,
            time_order=p.nu_or_tht, l=p.l_space,
            tol=1e-6, maxIt=50,
        )
        errors_grad.append(E_grad)

    return errors_grad


# Tests ___________________________________________________________________________________________________________________________________
def test_dg_spatial_convergence():
    solver_class = make_solver_class(SpaceMethod.DG, TimeMethod.BDF, DgParams())
    _assert_monotone_decrease(_collect_errors(solver_class), "DG-BDF")


def test_ldg_spatial_convergence():
    solver_class = make_solver_class(SpaceMethod.LDG, TimeMethod.BDF, LdgParams())
    _assert_monotone_decrease(_collect_errors(solver_class), "LDG-BDF")


def test_ppdg_spatial_convergence():
    solver_class = make_solver_class(SpaceMethod.PPDG, TimeMethod.BDF, PpDgParams())
    _assert_monotone_decrease(_collect_errors(solver_class), "PP-DG-BDF")


def test_spldg_spatial_convergence():
    solver_class = make_solver_class(SpaceMethod.SPLDG, TimeMethod.BDF, SpLdgParams())
    _assert_monotone_decrease(_collect_errors(solver_class), "SP-LDG-BDF")


def test_spldg_red2_spatial_convergence():
    solver_class = make_solver_class(SpaceMethod.SPLDG, TimeMethod.BDF, SpLdgParams(), full=False)
    _assert_monotone_decrease(_collect_errors(solver_class), "SP-LDG-BDF-REDUCED2")