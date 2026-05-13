from Utilities.EnumUtilities import SpaceMethod

# Dictionary for error labels 
ERROR_LABELS = {
    SpaceMethod.LDG:   ("E_c",  "E_q"),
    SpaceMethod.SPLDG: ("E_c",  "E_sigma"),
    SpaceMethod.PPDG:  ("E_L2", "E_DG"),
}

# Dictionary for variable labels
VARIABLE_LABELS = {
    SpaceMethod.LDG:   ("c_h", "q_h",     "D·∇c_ex"),
    SpaceMethod.SPLDG: ("c_h", "sigma_h", "-∇c_ex"),
    SpaceMethod.PPDG:  ("c_h", "c_h",     None)
}

# Dictionary for norm labels
NORM_LABELS = {
    SpaceMethod.LDG:   ("L2", "L2"),
    SpaceMethod.SPLDG: ("L2", "L2"),
    SpaceMethod.PPDG:  ("L2", "DG")
}