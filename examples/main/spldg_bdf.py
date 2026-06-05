"""Convergence launcher — SpLdgBDF."""

from examples._run import launch

from fisher_kolmogorov.models.solver_spldg_bdf  import SolverSpLdgBDF
from fisher_kolmogorov.configs.model_configs    import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType

# Manual configuration ____________________________________________________________________________________________________________________
_PARAMS_CONFIG = {
    TestType.COSINE: SpLdgParams(eps=0.0, eta_0=1.0, theta=-1.0),
    TestType.WAVE:   SpLdgParams(eps=0.0, eta_0=1.0, theta=-1.0, smoothing=1e-12),
}
_CONV_TYPE     = ConvType.SPATIAL  # SPATIAL | POLYNOMIAL | TEMPORAL
_TEST_TYPE     = TestType.COSINE   # COSINE  | WAVE
_SOLVER_KWARGS = {}

# Main ____________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    launch(
        solver_class = SolverSpLdgBDF,
        model_params = _PARAMS_CONFIG[_TEST_TYPE],
        conv_type    = _CONV_TYPE,
        test_type    = _TEST_TYPE,
        **_SOLVER_KWARGS,
    )
