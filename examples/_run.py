"""
Shared launcher for examples/main scripts.

Import this module from any examples/main/{solver}.py and call ``launch``.
The correct TestConfig is built automatically via the factory registry,
based on whether the solver derives from SolverBDF, and on ConvType/TestType.
"""

from fisher_kolmogorov.models.solver_bdf        import SolverBDF
from fisher_kolmogorov.runners                  import (
    run_spatial_convergence,
    run_polynomial_convergence,
    run_temporal_convergence,
)
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType
from fisher_kolmogorov.utilities.print_utilities import print_title, print_subtitle

from fisher_kolmogorov.configs.test_configs.cosine import (
    make_cosine_spatial,
    make_cosine_polynomial,
    make_cosine_temporal,
)
from fisher_kolmogorov.configs.test_configs.wave import (
    make_wave_spatial,
    make_wave_polynomial,
    make_wave_temporal,
)

# (is_bdf, conv_type, test_type) -> factory callable
_FACTORY_REGISTRY = {
    (True,  ConvType.SPATIAL,    TestType.COSINE): make_cosine_spatial,
    (False, ConvType.SPATIAL,    TestType.COSINE): make_cosine_spatial,
    (True,  ConvType.POLYNOMIAL, TestType.COSINE): make_cosine_polynomial,
    (False, ConvType.POLYNOMIAL, TestType.COSINE): make_cosine_polynomial,
    (True,  ConvType.TEMPORAL,   TestType.COSINE): make_cosine_temporal,
    (False, ConvType.TEMPORAL,   TestType.COSINE): make_cosine_temporal,
    (True,  ConvType.SPATIAL,    TestType.WAVE):   make_wave_spatial,
    (False, ConvType.SPATIAL,    TestType.WAVE):   make_wave_spatial,
    (True,  ConvType.POLYNOMIAL, TestType.WAVE):   make_wave_polynomial,
    (False, ConvType.POLYNOMIAL, TestType.WAVE):   make_wave_polynomial,
    (True,  ConvType.TEMPORAL,   TestType.WAVE):   make_wave_temporal,
    (False, ConvType.TEMPORAL,   TestType.WAVE):   make_wave_temporal,
}

_RUNNER_MAP = {
    ConvType.SPATIAL:    (run_spatial_convergence,    "Space convergence"),
    ConvType.POLYNOMIAL: (run_polynomial_convergence, "Polynomial degree convergence"),
    ConvType.TEMPORAL:   (run_temporal_convergence,   "Time convergence"),
}


def launch(solver_class, model_params, conv_type, test_type,
           tol=1e-11, max_it=200, **solver_kwargs):
    """
    Run a single convergence study for the given solver and model parameters.

    Parameters
    ----------
    solver_class  : type
        One of the 8 concrete solver classes.
    model_params  : ModelParams
        Matching parameter dataclass instance (e.g. LdgParams(C11=1.0)).
    conv_type     : ConvType
        Study to run: SPATIAL, POLYNOMIAL, or TEMPORAL.
    test_type     : TestType
        Physical test case: COSINE or WAVE.
    tol           : float
        Nonlinear solver tolerance.
    max_it        : int
        Maximum nonlinear iterations.
    **solver_kwargs
        Additional solver-specific keyword arguments passed directly to the runner.
    """
    is_bdf           = issubclass(solver_class, SolverBDF)
    factory          = _FACTORY_REGISTRY[(is_bdf, conv_type, test_type)]
    config           = factory()
    runner, subtitle = _RUNNER_MAP[conv_type]

    print_title(f"{solver_class.__name__}  ·  {config.name}")
    print_subtitle(subtitle)
    runner(
        solver_class = solver_class,
        config       = config,
        model_params = model_params,
        tol          = tol,
        max_it       = max_it,
        **solver_kwargs,
    )
