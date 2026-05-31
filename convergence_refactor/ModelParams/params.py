from dataclasses import dataclass

from ModelParams.base import ModelParams


# DG _______________________________________________________________________________________________________________________________________
from utilities.enum_utilities import PenaltyType

@dataclass
class DgParams(ModelParams):
    """
    Parameters for SolverDgBDF / SolverDgTheta.

    Attributes:
        eta_0 : interior-penalty coefficient.
        theta : penalty type (SIP=1, IIP=0, NIP=-1).
    """
    eta_0: float       = 10.0
    gamma: PenaltyType = PenaltyType.SIP

    def to_kwargs(self) -> dict:
        return {"eta_0": self.eta_0, "gamma": self.gamma}


# LDG ______________________________________________________________________________________________________________________________________
@dataclass
class LdgParams(ModelParams):
    """
    Parameters for SolverLdgBDF / SolverLdgTheta.

    Attributes:
        C11: stabilisation coefficient(default = 10.0).
        C12: normal flux coefficient(default = 0.5).
    """
    C11: float = 10.0
    C12: float = 0.5

    def to_kwargs(self) -> dict:
        return {"C11": self.C11, "C12": self.C12}
    

# PPDG ____________________________________________________________________________________________________________________________________
@dataclass
class PpDgParams(ModelParams):
    """
    Parameters for SolverPpDgBDF / SolverPpDgTheta.

    Attributes:
        eps      : stabilisation coeficient(useless).
        eta_0    : penalty coefficient(default = 10.0).
        smoothing: smoothing parameter for the positive-part operator(default = 0.0).
    """
    eps      : float = 0.0
    eta_0    : float = 10.0
    smoothing: float = 0.0

    def to_kwargs(self) -> dict:
        return {"eps": self.eps, "eta_0": self.eta_0, "smoothing": self.smoothing}


# SPLDG ___________________________________________________________________________________________________________________________________
@dataclass
class SpLdgParams(ModelParams):
    """
    Parameters for SolverSpLdgBDF / SolverSpLdgTheta.

    Attributes:
        eps  : LDG form stabilisation coefficient(default = 0.0).
        eta_0: penalty coefficient(default 1.0).
        theta: SP-LDG averaging exponent(default = -1.0).
    """
    eps  : float = 0.0
    eta_0: float = 1.0
    theta: float = -1.0

    def to_kwargs(self) -> dict:
        return {"eps": self.eps, "eta_0": self.eta_0, "theta": self.theta}
