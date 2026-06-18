"""Convergence launcher — DgTheta."""

from examples._run import launch

from fisher_kolmogorov.models.solver_factory    import make_solver_class
from fisher_kolmogorov.configs.model_configs    import DgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType, SpaceMethod, TimeMethod, PolyDegree

# Manual configuration ____________________________________________________________________________________________________________________
_PARAMS_CONFIG = {
    TestType.COSINE: DgParams(eta_0=10.0, linearize=False),
    TestType.WAVE:   DgParams(eta_0=10.0, linearize=False),
}
_CONV_TYPE     = ConvType.TEMPORAL  # SPATIAL | POLYNOMIAL | TEMPORAL
_TEST_TYPE     = TestType.COSINE   # COSINE  | WAVE

# Main ____________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    params       = _PARAMS_CONFIG[_TEST_TYPE]
    solver_class = make_solver_class(
        space     = SpaceMethod.DG,
        time      = TimeMethod.THETA,
        params    = params,
    )
    launch(
        solver_class   = solver_class,
        conv_type      = _CONV_TYPE,
        test_type      = _TEST_TYPE,
        nu_or_tht      = 0.5,
        factory_kwargs = {"l_space": PolyDegree.P4}
    )
