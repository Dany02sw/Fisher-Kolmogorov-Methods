from enum import Enum, auto, IntEnum


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
    DG    = auto()
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

class PolyDegree(IntEnum):
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5
    P6 = 6
    P7 = 7
    P8 = 8

class BdfOrder(IntEnum):
    BDF1 = 1
    BDF2 = 2
    BDF3 = 3
    BDF4 = 4
    BDF5 = 5
    BDF6 = 6

class ThetaMethod(float, Enum):
    EE = 0.0
    CN = 0.5
    IE = 1.0

class PenaltyType(float, Enum):
    SIP =   1.0
    NIP = - 1.0
    IIP =   0.0