from dataclasses import dataclass


@dataclass
class ModelParams:
    """
    Base class for model-specific parameters.

    Subclasses add the fields that are meaningful for their solver
    (e.g. stabilisation coefficients, penalty parameters).
    Each subclass must implement ``to_kwargs``, which returns a dict
    suitable for unpacking into the solver constructor.
    """

    def to_kwargs(self) -> dict:
        """Return a dict of keyword arguments for the solver constructor."""
        raise NotImplementedError
