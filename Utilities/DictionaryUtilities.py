from Utilities.EnumUtilities import SpaceMethod

# Dictionary for error labels 
ERROR_LABELS = {
    SpaceMethod.LDG:   ("E_c",  "E_q"),
    SpaceMethod.SPLDG: ("E_c",  "E_sigma"),
    SpaceMethod.PPDG:  ("E_L2", "E_DG"),
}

# Dictionary for variable labels
VARIABLE_LABELS = {
    SpaceMethod.LDG:   ("c", "q",     "D·∇c"),
    SpaceMethod.SPLDG: ("c", "sigma", "-∇c"),
    SpaceMethod.PPDG:  ("c", "c",     None)
}

# Dictionary for norm labels
NORM_LABELS = {
    SpaceMethod.LDG:   ("L2", "L2"),
    SpaceMethod.SPLDG: ("L2", "L2"),
    SpaceMethod.PPDG:  ("L2", "DG")
}