from enum import Enum, auto

# Enum for convergence type
class ConvType(Enum):
    SPATIAL    = auto()
    POLYNOMIAL = auto()
    TEMPORAL   = auto()

# Enum for analysis type
class AnalysisType(Enum):
    RATE       = auto() 
    SATURATION = auto()

# Enum for brain sections mesh
class BrainSection(Enum):
    SAGITTAL   = auto()
    CORONAL    = auto()
    HORIZONTAL = auto()