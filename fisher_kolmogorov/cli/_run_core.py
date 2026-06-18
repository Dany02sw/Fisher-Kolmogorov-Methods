"""
Shared convergence launcher — core logic used by both the CLI (fk-convergence)
and the examples/_run.py script.

Public API
----------
launch(solver_class, conv_type, test_type, ...)
    Run a single convergence study.

FACTORY_REGISTRY
    Maps (ConvType, TestType) to the correct factory callable.

RUNNER_MAP
    Maps ConvType to (runner_callable, subtitle_string).
"""


from fisher_kolmogorov.runners import (
    run_spatial_convergence,
    run_polynomial_convergence,
    run_temporal_convergence,
)
from fisher_kolmogorov.utilities.enum_utilities  import ConvType, TestType, BdfOrder, PolyDegree, ThetaMethod
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

# (conv_type, test_type) -> factory callable
# All time methods share the same convergence test factories
FACTORY_REGISTRY = {
    (ConvType.SPATIAL,    TestType.COSINE): make_cosine_spatial,
    (ConvType.POLYNOMIAL, TestType.COSINE): make_cosine_polynomial,
    (ConvType.TEMPORAL,   TestType.COSINE): make_cosine_temporal,
    (ConvType.SPATIAL,    TestType.WAVE):   make_wave_spatial,
    (ConvType.POLYNOMIAL, TestType.WAVE):   make_wave_polynomial,
    (ConvType.TEMPORAL,   TestType.WAVE):   make_wave_temporal,
}

RUNNER_MAP = {
    ConvType.SPATIAL:    (run_spatial_convergence,    "Space convergence"),
    ConvType.POLYNOMIAL: (run_polynomial_convergence, "Polynomial degree convergence"),
    ConvType.TEMPORAL:   (run_temporal_convergence,   "Time convergence"),
}

def _coerce_nu_or_tht(val):
    """
    Normalize a raw or already-typed time-method value to BdfOrder or ThetaMethod.

    Accepts:
        BdfOrder    → returned as-is
        ThetaMethod → returned as-is
        int         → converted to BdfOrder
        float       → converted to ThetaMethod

    Raises
    ------
    ValueError
        If an int is out of BdfOrder range or a float is not a valid ThetaMethod value.
    TypeError
        If val is none of the above.
    """
    if isinstance(val, (BdfOrder, ThetaMethod)):
        return val
    if isinstance(val, int):
        return BdfOrder(val)
    if isinstance(val, float):
        return ThetaMethod(val)
    raise TypeError(
        f"nu_or_tht must be BdfOrder, ThetaMethod, int, or float; got {type(val).__name__!r}"
    )



def launch(solver_class, conv_type, test_type,
           tol=1e-11, max_it=200, l=None, nu_or_tht=None,
           factory_kwargs: dict = None, output_dir=None, save_plot=False):
    """
    Run a single convergence study for the given solver and model parameters.

    Parameters
    ----------
    solver_class : type
        A concrete solver class (or one built by make_solver_class).
    conv_type    : ConvType
        Study to run: SPATIAL, POLYNOMIAL, or TEMPORAL.
    test_type    : TestType
        Physical test case: COSINE or WAVE.
    tol          : float
        Nonlinear solver tolerance.
    max_it       : int
        Maximum nonlinear iterations.
    l            : int, optional
        Polynomial degree (e.g. 2 for P2); converted to PolyDegree and
        forwarded as ``l_space`` for SPATIAL and TEMPORAL studies.
        Ignored for POLYNOMIAL studies (the factory iterates over ``l_list``).
    nu_or_tht    : BdfOrder | ThetaMethod | int | float | None
        Time integration parameter. Accepts raw values (int → BdfOrder,
        float → ThetaMethod) or already-typed enums.
        If None, the factory uses its own default.
    factory_kwargs : dict, optional
        Additional keyword arguments forwarded directly to the factory,
        overriding any value derived from ``l`` or ``nu_or_tht``.
    output_dir   : Path or None, optional
        If provided, solver solutions are written to XDMF files in this
        directory (only the polynomial runner currently supports it).
    save_plot    : bool
        If True, convergence plots are saved to disk.
    """
    factory = FACTORY_REGISTRY[(conv_type, test_type)]

    resolved = {}
    if nu_or_tht is not None:
        resolved["nu_or_tht"] = _coerce_nu_or_tht(nu_or_tht)
    if l is not None and conv_type != ConvType.POLYNOMIAL:
        resolved["l_space"] = PolyDegree(l)
    if factory_kwargs:
        resolved.update(factory_kwargs)

    config           = factory(**resolved)
    runner, subtitle = RUNNER_MAP[conv_type]

    parts = [solver_class.__name__]
    if l         is not None: parts.append(f"l={l}")
    if nu_or_tht is not None: parts.append(repr(_coerce_nu_or_tht(nu_or_tht)))
    print_title(" · ".join(parts))
    print_subtitle(subtitle)
    runner_kwargs = dict(
        solver_class = solver_class,
        config       = config,
        tol          = tol,
        max_it       = max_it,
        save_plot    = save_plot,
    )
    if output_dir is not None:
        runner_kwargs["output_dir"] = output_dir
    runner(**runner_kwargs)