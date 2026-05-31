from dolfin import *

from fisher_kolmogorov.configs.test_configs.base        import BrainConfig
from fisher_kolmogorov.utilities.enum_utilities import BrainSection

# Configuration for sagittal section simulation ___________________________________________________________________________________________
BRAIN_SAGITTAL = BrainConfig(
    plane       = BrainSection.SAGITTAL,
    alpha_grey  = 0.5,
    alpha_white = 1.0,
    d_ext       = 8e-3,
    d_axn       = 8e-2,
    a           = (1.0, 1.0),
)

# Configuration for coronal section simulation ____________________________________________________________________________________________
BRAIN_CORONAL = BrainConfig(
    plane       = BrainSection.CORONAL,
    alpha_grey  = 0.5,
    alpha_white = 1.0,
    d_ext       = 8e-3,
    d_axn       = 8e-2,
    a           = (1.0, 1.0),
)

# Configuration for horizzontal section simulation _________________________________________________________________________________________
BRAIN_HORIZONTAL = BrainConfig(
    plane       = BrainSection.HORIZONTAL,
    alpha_grey  = 0.5,
    alpha_white = 1.0,
    d_ext       = 8e-3,
    d_axn       = 8e-2,
    a           = (1.0, 1.0),
)