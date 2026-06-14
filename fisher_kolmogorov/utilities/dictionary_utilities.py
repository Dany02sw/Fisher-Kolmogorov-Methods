from fisher_kolmogorov.utilities.enum_utilities import SpaceMethod, BrainSection
from fisher_kolmogorov.config import SAGITTAL_DIR, CORONAL_DIR, HORIZONTAL_DIR

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
    SpaceMethod.SPLDG: (r"$||c_{\rm ex}(\cdot,T) - u(w_h^{(N)})||_{L^2(\Omega)}$",         r"$||\nabla c_{\rm ex}(\cdot,T) + \mathbf{\sigma}_h^{(N)}||_{L^2(\Omega)}$"),
    SpaceMethod.PPDG:  (r"$||c_{\rm ex}(\cdot,T) - e^{\lambda_h^{(N)}}||_{L^2(\Omega)}$",  r"$||c_{\rm ex}(\cdot,T) - e^{\lambda_h^{(N)}}||_{DG}$"),
}

# Dictionary to match directories for brain
BRAIN_SECTION_DIRS = {
    BrainSection.SAGITTAL:   SAGITTAL_DIR,
    BrainSection.CORONAL:    CORONAL_DIR,
    BrainSection.HORIZONTAL: HORIZONTAL_DIR,
}