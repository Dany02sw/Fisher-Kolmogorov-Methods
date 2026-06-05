from fisher_kolmogorov.utilities.enum_utilities import SpaceMethod, TimeMethod

from fisher_kolmogorov.plots.models_config import (
    dg_bdf, dg_theta,
    ldg_bdf, ldg_theta,
    spldg_bdf, spldg_theta,
    ppdg_bdf, ppdg_theta,
)

CONFIG_REGISTRY = {
    (SpaceMethod.DG,    TimeMethod.BDF):   dg_bdf,
    (SpaceMethod.DG,    TimeMethod.THETA): dg_theta,
    (SpaceMethod.LDG,   TimeMethod.BDF):   ldg_bdf,
    (SpaceMethod.LDG,   TimeMethod.THETA): ldg_theta,
    (SpaceMethod.SPLDG, TimeMethod.BDF):   spldg_bdf,
    (SpaceMethod.SPLDG, TimeMethod.THETA): spldg_theta,
    (SpaceMethod.PPDG,  TimeMethod.BDF):   ppdg_bdf,
    (SpaceMethod.PPDG,  TimeMethod.THETA): ppdg_theta,
}


def load_config(space: SpaceMethod, time: TimeMethod) -> dict:
    """
    Return the data namespace for the given (space, time) combination
    as a plain dict, ready for globals().update().

    Raises KeyError with a descriptive message if the combination has
    no associated config module.
    """
    key = (space, time)
    if key not in CONFIG_REGISTRY:
        available = ", ".join(f"({s.name}, {t.name})" for s, t in CONFIG_REGISTRY)
        raise KeyError(
            f"No config for ({space.name}, {time.name}). "
            f"Available combinations: {available}"
        )
    mod = CONFIG_REGISTRY[key]
    return {k: v for k, v in vars(mod).items() if not k.startswith("_")}