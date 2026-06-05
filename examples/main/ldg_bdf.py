"""Convergence launcher — LdgBDF."""

from examples._run import launch

from fisher_kolmogorov.models.solver_ldg_bdf    import SolverLdgBDF
from fisher_kolmogorov.configs.model_configs    import LdgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType

# Manual configuration ____________________________________________________________________________________________________________________
_PARAMS_CONFIG = {
    TestType.COSINE: LdgParams(C11=1.0, C12=0.5),
    TestType.WAVE:   LdgParams(C11=1.0, C12=0.5),
}
_CONV_TYPE     = ConvType.SPATIAL  # SPATIAL | POLYNOMIAL | TEMPORAL
_TEST_TYPE     = TestType.COSINE   # COSINE  | WAVE
_SOLVER_KWARGS = {"Linearize": True}

# Main ____________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    launch(
        solver_class = SolverLdgBDF,
        model_params = _PARAMS_CONFIG[_TEST_TYPE],
        conv_type    = _CONV_TYPE,
        test_type    = _TEST_TYPE,
        **_SOLVER_KWARGS,
    )
