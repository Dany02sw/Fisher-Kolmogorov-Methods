"""
Shared convergence launcher — core logic used by both the CLI (fk-convergence)
and the examples/_run.py script.

Public API
----------
launch(solver_class, model_params, conv_type, test_type, ...)
    Run a single convergence study.

FACTORY_REGISTRY
    Maps (is_bdf, ConvType, TestType) to the correct factory callable.

RUNNER_MAP
    Maps ConvType to (runner_callable, subtitle_string).
"""

from fisher_kolmogorov.models.solver_bdf        import SolverBDF
from fisher_kolmogorov.runners                  import (
    run_spatial_convergence,
    run_polynomial_convergence,
    run_temporal_convergence,
)
from fisher_kolmogorov.utilities.enum_utilities  import ConvType, TestType, BdfOrder, PolyDegree
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
FACTORY_REGISTRY = {
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

RUNNER_MAP = {
    ConvType.SPATIAL:    (run_spatial_convergence,    "Space convergence"),
    ConvType.POLYNOMIAL: (run_polynomial_convergence, "Polynomial degree convergence"),
    ConvType.TEMPORAL:   (run_temporal_convergence,   "Time convergence"),
}


def launch(solver_class, model_params, conv_type, test_type,
           tol=1e-11, max_it=200, l=None, nu=None,
           factory_kwargs: dict = None, output_dir=None, **solver_kwargs):
    """
    Run a single convergence study for the given solver and model parameters.

    Parameters
    ----------
    solver_class    : type
        One of the concrete solver classes.
    model_params    : ModelParams
        Matching parameter dataclass instance (e.g. LdgParams(C11=1.0)).
    conv_type       : ConvType
        Study to run: SPATIAL, POLYNOMIAL, or TEMPORAL.
    test_type       : TestType
        Physical test case: COSINE or WAVE.
    tol             : float
        Nonlinear solver tolerance.
    max_it          : int
        Maximum nonlinear iterations.
    l               : int, optional
        Polynomial degree integer (e.g. 2 for P2); converted to PolyDegree
        and forwarded to the factory as ``l_space`` for SPATIAL and TEMPORAL
        studies. Ignored for POLYNOMIAL studies, whose factory iterates over
        an internal ``l_list`` and does not accept a single ``l_space``.
    nu              : int, optional
        BDF order integer (e.g. 3 for BDF3); converted to BdfOrder and
        forwarded to the factory as ``nu_or_tht`` for all study types.
        Theta-method solvers (``is_bdf=False``) never pass ``nu``, so this
        stays None and the factory falls back to its default ThetaMethod.
    factory_kwargs  : dict, optional
        Additional keyword arguments forwarded directly to the factory,
        overriding any value derived from ``l`` or ``nu``. Used by the CLI
        to pass mesh sizes, dt_list, l_list, config_kwargs, and so on.
    output_dir      : Path or None, optional
        If provided, solver solutions are written to XDMF files in this
        directory. Forwarded to the runner; only the polynomial runner
        currently supports it (used for model comparison studies).
    **solver_kwargs
        Additional keyword arguments passed directly to the runner.
    """
    is_bdf  = issubclass(solver_class, SolverBDF)
    factory = FACTORY_REGISTRY[(is_bdf, conv_type, test_type)]

    resolved = {}
    if nu is not None:
        resolved["nu_or_tht"] = BdfOrder(nu)
    if l is not None and conv_type != ConvType.POLYNOMIAL:
        resolved["l_space"] = PolyDegree(l)
    if factory_kwargs:
        resolved.update(factory_kwargs)

    config           = factory(**resolved)
    runner, subtitle = RUNNER_MAP[conv_type]

    parts = [solver_class.__name__]
    if l  is not None: parts.append(f"l={l}")
    if nu is not None: parts.append(f"BDF{nu}")
    print_title(" · ".join(parts))
    print_subtitle(subtitle)
    runner_kwargs = dict(
        solver_class = solver_class,
        config       = config,
        model_params = model_params,
        tol          = tol,
        max_it       = max_it,
        **solver_kwargs,
    )
    if output_dir is not None:
        runner_kwargs["output_dir"] = output_dir
    runner(**runner_kwargs)
