"""Shared CLI registries: solver keys, space/time methods, and default ModelParams."""

from fisher_kolmogorov.utilities.enum_utilities import SpaceMethod, TimeMethod, ThetaMethod, BdfOrder
from fisher_kolmogorov.configs.model_configs    import DgParams, LdgParams, PpDgParams, SpLdgParams

# Maps CLI solver key -> (SpaceMethod, TimeMethod, is_full_spldg)
_SOLVER_REGISTRY = {
    "dg_bdf"           : (SpaceMethod.DG,    TimeMethod.BDF,   False),
    "dg_theta"         : (SpaceMethod.DG,    TimeMethod.THETA, False),
    "ldg_bdf"          : (SpaceMethod.LDG,   TimeMethod.BDF,   False),
    "ldg_theta"        : (SpaceMethod.LDG,   TimeMethod.THETA, False),
    "ppdg_bdf"         : (SpaceMethod.PPDG,  TimeMethod.BDF,   False),
    "ppdg_theta"       : (SpaceMethod.PPDG,  TimeMethod.THETA, False),
    "spldg_bdf"        : (SpaceMethod.SPLDG, TimeMethod.BDF,   False),
    "spldg_theta"      : (SpaceMethod.SPLDG, TimeMethod.THETA, False),
    "spldg_bdf_full"   : (SpaceMethod.SPLDG, TimeMethod.BDF,   True),
    "spldg_theta_full" : (SpaceMethod.SPLDG, TimeMethod.THETA, True),
}

# Default ModelParams class per solver key
_PARAMS_DEFAULTS = {
    "dg_bdf"         : DgParams,    "dg_theta"         : DgParams,
    "ldg_bdf"        : LdgParams,   "ldg_theta"        : LdgParams,
    "ppdg_bdf"       : PpDgParams,  "ppdg_theta"       : PpDgParams,
    "spldg_bdf"      : SpLdgParams, "spldg_theta"      : SpLdgParams,
    "spldg_bdf_full" : SpLdgParams, "spldg_theta_full" : SpLdgParams,
}

# Time method registry
_SCHEME_MAP  = {
    **{f"bdf{o.value}": o for o in BdfOrder},
    "ie": ThetaMethod.IE,
    "cn": ThetaMethod.CN,
    "ee": ThetaMethod.EE,
}