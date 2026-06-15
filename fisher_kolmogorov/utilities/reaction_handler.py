from abc    import ABC, abstractmethod
from typing import Union
from numpy  import exp
from dolfin import Constant
import ufl

from fisher_kolmogorov.utilities.enum_utilities import SpaceMethod, ThetaMethod

# Reaction handler base class _______________________________________________
class ReactionHandler(ABC):

    def __init__(self, alpha: Union[float, Constant]):
        self.alpha = float(alpha) if isinstance(alpha, (Constant, ufl.core.expr.Expr)) else alpha
        self.tau   = None

    def _compute_exact(self, c0):
        """Exact solution of the logistic ODE u' = alpha*u*(1-u) over interval tau."""
        c0_np    = c0.vector()[:]
        exp_term = exp(self.alpha * self.tau) 
        return (c0_np*exp_term)/ (1.0 - c0_np + exp_term)

    @abstractmethod
    def pre_step(self, t0: float, c0) -> None: ...

    @abstractmethod
    def post_step(self, t0: float, c0) -> None: ...


# Lie splitting handler _____________________________________________________
class LieHandler(ReactionHandler):
    """Reaction handler for Lie (Godunov) splitting. Solves reaction exactly over dt before diffusion."""

    def __init__(self, alpha: Union[float, Constant], dt: float):
        super().__init__(alpha)
        self.tau = dt

    def pre_step(self, t0: float, c0) -> None:
        c0.vector()[:] = self._compute_exact(c0)

    def post_step(self, t0: float, c0) -> None:
        pass


# Strang splitting handler __________________________________________________
class StrangHandler(ReactionHandler):
    """Reaction handler for Strang splitting. Solves reaction exactly over dt/2 before and after diffusion."""

    def __init__(self, alpha: Union[float, Constant], dt: float):
        super().__init__(alpha)
        self.tau = dt / 2.0

    def pre_step(self, t0: float, c0) -> None:
        c0.vector()[:] = self._compute_exact(c0)

    def post_step(self, t0: float, c0) -> None:
        c0.vector()[:] = self._compute_exact(c0)


# Null handler for structure-preserving methods _____________________________
class NullHandler(ReactionHandler):
    """No-op handler for structure-preserving methods, which do not support splitting."""

    def __init__(self):
        pass  # no alpha or tau needed

    def pre_step(self, t0: float, c0) -> None:
        pass

    def post_step(self, t0: float, c0) -> None:
        pass


# Reaction handler factory __________________________________________________
def make_reaction_handler(
    SM    : SpaceMethod,
    tht   : Union[ThetaMethod.IE, ThetaMethod.CN],
    alpha : Union[float, Constant],
    dt    : float
) -> ReactionHandler:
    """
    Factory for reaction handlers. Used only within the theta-method time mixin.

    - SP methods (SPLDG, PPDG) -> NullHandler
    - Non-SP + IE (theta=1)    -> LieHandler
    - Non-SP + CN (theta=0.5)  -> StrangHandler
    """
    if SM in (SpaceMethod.DG, SpaceMethod.LDG):
        if tht == ThetaMethod.IE:
            return LieHandler(alpha, dt)
        elif tht == ThetaMethod.CN:
            return StrangHandler(alpha, dt)
    return NullHandler()