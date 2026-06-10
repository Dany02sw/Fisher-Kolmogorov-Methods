"""Convergence launcher — LDG + BDF."""

from examples._run import launch

from fisher_kolmogorov.models.solver_factory    import make_solver_class
from fisher_kolmogorov.configs.model_configs    import LdgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType, SpaceMethod, TimeMethod

# Manual configuration ____________________________________________________________________________________________________________________
_PARAMS_CONFIG = {
    TestType.COSINE: LdgParams(C11=1.0, C12=0.5),
    TestType.WAVE:   LdgParams(C11=1.0, C12=0.5),
}
_CONV_TYPE  = ConvType.SPATIAL  # SPATIAL | POLYNOMIAL | TEMPORAL
_TEST_TYPE  = TestType.COSINE   # COSINE  | WAVE
_LINEARIZE  = True

# Main ____________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    params       = _PARAMS_CONFIG[_TEST_TYPE]
    solver_class = make_solver_class(
        space     = SpaceMethod.LDG,
        time      = TimeMethod.BDF,
        params    = params,
        linearize = _LINEARIZE,
    )
    launch(
        solver_class = solver_class,
        model_params = params,
        conv_type    = _CONV_TYPE,
        test_type    = _TEST_TYPE,
    )
