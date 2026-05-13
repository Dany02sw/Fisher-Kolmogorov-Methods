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

# Enum for time discretization type
class TimeMethod(Enum):
    BDF   = auto()
    THETA = auto()

# Enum for the space discretization
class SpaceMethod(Enum):
    LDG   = auto()
    PPDG  = auto()
    SPLDG = auto()

# Enum for transformations
class Transformations(Enum):
    IDENTITY    = auto()
    EXPONENTIAL = auto()
    ENTROPIC    = auto()

class MeshType(Enum):
    UNIT_SQUARE  = auto()
    RECTANGLE    = auto()
    BRAIN_2D     = auto()

class MeshStructure(Enum):
    STRUCTURED   = auto()
    UNSTRUCTURED = auto()