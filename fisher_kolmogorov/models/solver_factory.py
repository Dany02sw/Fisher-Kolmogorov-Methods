"""
Factory for building solver classes by composing space and time mixins.

Usage
-----
    SolverDgBdf = make_solver_class(
        space  = SpaceMethod.DG,
        time   = TimeMethod.BDF,
        params = DgParams(eta_0=10.0),
    )
    solver = SolverDgBdf(mesh, D, alpha, c_0)
    solver.Solve(t0, dt, T, time_order=2, ...)

    # With optional solver kwargs (e.g. linearize, only meaningful for BDF):
    SolverDgBdf = make_solver_class(
        space     = SpaceMethod.DG,
        time      = TimeMethod.BDF,
        params    = DgParams(eta_0=10.0),
        linearize = True,
    )

Space methods and their ModelParams
------------------------------------
    SpaceMethod.DG    → DgParams(eta_0, gamma=PenaltyType.SIP)
    SpaceMethod.LDG   → LdgParams(C11, C12)
    SpaceMethod.PPDG  → PpDgParams(eps, eta_0, smoothing=0.0)
    SpaceMethod.SPLDG → SpLdgParams(eps, eta_0, theta, smoothing=0.0)

    SpaceMethod.SPLDG defaults to the reduced (2-component) formulation.
    Pass full=True to use the full (4-component) formulation instead.

Note: 'linearize=True' is silently ignored when time=TimeMethod.THETA,
because explicit extrapolation is only defined for BDF schemes.
"""

import warnings

from fisher_kolmogorov.models.solver_base  import SolverBase

from fisher_kolmogorov.models.space_mixins import (
    SpaceMixinDg, SpaceMixinLdg,
    SpaceMixinPpDg,
    SpaceMixinSpLdgReduced, SpaceMixinSpLdg,
)
from fisher_kolmogorov.models.time_mixins         import TimeMixinBdf, TimeMixinTheta

from fisher_kolmogorov.utilities.enum_utilities   import SpaceMethod, TimeMethod
from fisher_kolmogorov.configs.model_configs.base import ModelParams


_SPACE_MIXINS = {
    SpaceMethod.DG:    SpaceMixinDg,
    SpaceMethod.LDG:   SpaceMixinLdg,
    SpaceMethod.PPDG:  SpaceMixinPpDg,
    SpaceMethod.SPLDG: SpaceMixinSpLdgReduced,  # default; overridden by full=True
}

_TIME_MIXINS = {
    TimeMethod.BDF:   TimeMixinBdf,
    TimeMethod.THETA: TimeMixinTheta,
}


def make_solver_class(
    space     : SpaceMethod,
    time      : TimeMethod,
    params    : ModelParams,
    full      : bool = False,
) -> type:
    """Build and return a solver class for the given space/time combination.

    The returned class has the same interface as the concrete solver classes
    it replaces: instantiate with (mesh, D, alpha, c_0), then call .Solve()
    or .ConvergenceTest().

    Parameters
    ----------
    space     : SpaceMethod
        Spatial discretization to use.
    time      : TimeMethod
        Time discretization to use.
    params    : ModelParams
        Space-specific parameters (DgParams, LdgParams, PpDgParams, SpLdgParams).
    full      : bool
        Use the full 4-component SpLDG formulation instead of the reduced
        2-component one. Only relevant when space=SpaceMethod.SPLDG.

    Returns
    -------
    type
        A solver class ready to be instantiated with (mesh, D, alpha, c_0).
    """
    if space not in _SPACE_MIXINS:
        raise ValueError(f"Unknown space method: {space}")
    if time not in _TIME_MIXINS:
        raise ValueError(f"Unknown time method: {time}")

    space_mixin  = SpaceMixinSpLdg if (space is SpaceMethod.SPLDG and full) else _SPACE_MIXINS[space]
    time_mixin   = _TIME_MIXINS[time]
    space_kwargs = params.to_kwargs()

    # linearize is only meaningful for BDF; warn and strip if paired with Theta
    if time is TimeMethod.THETA and space_kwargs.get("linearize", False):
        warnings.warn(
            "'linearize=True' is not supported with TimeMethod.THETA and will be ignored.",
            UserWarning, stacklevel=2,
        )

    # MRO: CombinedSolver -> space_mixin -> time_mixin -> SolverBase
    space_tag = space_mixin.__name__.replace("SpaceMixin", "")
    time_tag  = time_mixin.__name__.replace("TimeMixin",  "")

    class Solver(space_mixin, time_mixin, SolverBase):

        def __init__(self, mesh, D, alpha, c_0):
            SolverBase.__init__(self, mesh, D, alpha, c_0)
            self._init_time()
            self._init_space(**space_kwargs)

    Solver.__name__     = f"Solver{space_tag}{time_tag}"
    Solver.__qualname__ = Solver.__name__

    return Solver
