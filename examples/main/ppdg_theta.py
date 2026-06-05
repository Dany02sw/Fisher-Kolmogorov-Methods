"""Convergence launcher — PpDgTheta."""

from examples._run import launch

from fisher_kolmogorov.models.solver_ppdg_theta import SolverPpDgTheta
from fisher_kolmogorov.configs.model_configs    import PpDgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType

# Manual configuration ____________________________________________________________________________________________________________________
_PARAMS_CONFIG = {
    TestType.COSINE: PpDgParams(eps=0.0, eta_0=10.0, smoothing=0.0),
    TestType.WAVE:   PpDgParams(eps=0.0, eta_0=10.0, smoothing=0.0),
}
_CONV_TYPE     = ConvType.SPATIAL  # SPATIAL | POLYNOMIAL | TEMPORAL
_TEST_TYPE     = TestType.COSINE   # COSINE  | WAVE
_SOLVER_KWARGS = {}

# Main ____________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    launch(
        solver_class = SolverPpDgTheta,
        model_params = _PARAMS_CONFIG[_TEST_TYPE],
        conv_type    = _CONV_TYPE,
        test_type    = _TEST_TYPE,
        **_SOLVER_KWARGS,
    )
