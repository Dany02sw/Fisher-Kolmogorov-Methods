from Utilities.EnumUtilities import SpaceMethod

# Dictionary for error labels 
ERROR_LABELS_PRINT = {
    SpaceMethod.DG:    ("E_L2", "E_DG"),
    SpaceMethod.LDG:   ("E_c",  "E_q"),
    SpaceMethod.SPLDG: ("E_c",  "E_sigma"),
    SpaceMethod.PPDG:  ("E_L2", "E_DG"),
}

ERROR_LABELS_PLOT = {
    SpaceMethod.DG:  ("E_{L²}", "E_{DG}"),
    SpaceMethod.LDG:   ("E_c",  "E_q"),
    SpaceMethod.SPLDG: ("E_c",  "E_σ"),
    SpaceMethod.PPDG:  ("E_{L²}", "E_{DG}"),
}

# Dictionary for norm labels
NORM_LABELS = {
    SpaceMethod.DG:  (r"$||c_{ex} - c_h||_{L^2(\Omega)}$",            r"$||c_{ex} - c_h||_{DG}$"),
    SpaceMethod.LDG:   (r"$||c_{ex} - c_h||_{L^2(\Omega)}$",            r"$||D\nabla c_{ex} - q_h||_{L^2(\Omega)}$"),
    SpaceMethod.SPLDG: (r"$||c_{ex} - u(w_h)||_{L^2(\Omega)}$",         r"$||\nabla c_{ex} + \sigma_h||_{L^2(\Omega)}$"),
    SpaceMethod.PPDG:  (r"$||c_{ex} - e^{\lambda_h}||_{L^2(\Omega)}$",  r"$||c_{ex} - e^{\lambda_h}||_{DG}$"),
}