"""Convergence launcher — DgTheta."""

from examples._run import launch

from fisher_kolmogorov.models.solver_dg_theta   import SolverDgTheta
from fisher_kolmogorov.configs.model_configs    import DgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType

# Manual configuration ____________________________________________________________________________________________________________________
_PARAMS_CONFIG = {
    TestType.COSINE: DgParams(eta_0=10.0),
    TestType.WAVE:   DgParams(eta_0=10.0),
}
_CONV_TYPE     = ConvType.SPATIAL  # SPATIAL | POLYNOMIAL | TEMPORAL
_TEST_TYPE     = TestType.COSINE   # COSINE  | WAVE
_SOLVER_KWARGS = {}

# Main ____________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    launch(
        solver_class = SolverDgTheta,
        model_params = _PARAMS_CONFIG[_TEST_TYPE],
        conv_type    = _CONV_TYPE,
        test_type    = _TEST_TYPE,
        **_SOLVER_KWARGS,
    )
