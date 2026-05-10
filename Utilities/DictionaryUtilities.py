from Utilities.EnumUtilities import SpaceMethod

# Dictionary for labels 
SPACE_LABELS = {
    SpaceMethod.LDG:   ("E_c", "E_q",     "D·∇c"),
    SpaceMethod.SPLDG: ("E_c", "E_sigma", "-∇c"),
    SpaceMethod.PPDG:  ("E_c", "E_DG",    None),
}

# Dictionary for variables
VARIABLE_LABELS = {
    SpaceMethod.LDG:   ("c", "q"),
    SpaceMethod.SPLDG: ("c", "sigma"),
    SpaceMethod.PPDG:  ("c", None)
}

# Dictionary for norms labels
NORM_LABELS = {
    SpaceMethod.LDG:   ("L2", "L2"),
    SpaceMethod.SPLDG: ("L2", "L2"),
    SpaceMethod.PPDG:  ("L2", "DG")
}