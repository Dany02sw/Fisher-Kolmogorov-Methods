from enum import Enum, auto


class ConvType(Enum):
    SPATIAL    = auto()
    POLYNOMIAL = auto()
    TEMPORAL   = auto()

class BrainSection(Enum):
    SAGITTAL   = auto()
    CORONAL    = auto()
    HORIZONTAL = auto()

class TimeMethod(Enum):
    BDF   = auto()
    THETA = auto()

class SpaceMethod(Enum):
    LDG   = auto()
    PPDG  = auto()
    SPLDG = auto()

class MeshType(Enum):
    UNIT_SQUARE  = auto()
    RECTANGLE    = auto()
    BRAIN_2D     = auto()

class MeshStructure(Enum):
    STRUCTURED   = auto()
    UNSTRUCTURED = auto()