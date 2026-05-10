from Utilities.EnumUtilities import SpaceMethod

# Dictionary for labels 
SPACE_LABELS = {
    SpaceMethod.LDG:   ("E_c", "E_q",     "D·∇c"),
    SpaceMethod.SPLDG: ("E_c", "E_sigma", "-∇c"),
    SpaceMethod.PPDG:  ("E_c", "E_DG",    "DG"),
}