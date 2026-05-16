from Utilities.EnumUtilities import SpaceMethod

# Dictionary for error labels 
ERROR_LABELS = {
    SpaceMethod.LDG:   ("E_c",  "E_q"),
    SpaceMethod.SPLDG: ("E_c",  "E_sigma"),
    SpaceMethod.PPDG:  ("E_L2", "E_DG"),
}

# Dictionary for norm labels
NORM_LABELS = {
    SpaceMethod.LDG:   ("‖cₑₓ  − cₕ‖_L²",      "‖D∇cₑₓ − qₕ‖_L²"),
    SpaceMethod.SPLDG: ("‖cₑₓ  − u(wₕ)‖_L²",   "‖∇cₑₓ + σₕ‖_L²"),
    SpaceMethod.PPDG:  ("‖cₑₓ  − exp(λₕ)‖_L²", "‖cₑₓ  − exp(λₕ)‖_DG")
}