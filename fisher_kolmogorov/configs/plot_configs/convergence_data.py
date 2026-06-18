from __future__ import annotations

from dataclasses import dataclass
from typing      import Dict, List, Optional

import numpy as np

from fisher_kolmogorov.utilities.enum_utilities import (
    PolyDegree, BdfOrder, ThetaMethod, TimeMethod,
)


@dataclass
class ConvergenceData:
    """Convergence and saturation results for a single solver configuration.

    All fields default to None; only populate the fields relevant to the
    studies you intend to plot. fk-plot raises AttributeError at dispatch
    time if a required field is missing for the requested plot type.

    Spatial convergence
    -------------------
    hs, errs_c_space, errs_grad_space

    Polynomial convergence
    ----------------------
    poly_conv_h, poly_conv_degrees, errs_c_poly, errs_grad_poly

    Temporal convergence
    --------------------
    dt, errs_c_bdf, errs_grad_bdf      (BDF)
    dt, errs_c_theta, errs_grad_theta  (Theta)

    Space saturation
    ----------------
    space_sat_hs, space_sat_degrees, space_sat_time_method,
    errs_c_space_sat, errs_grad_space_sat

    Polynomial saturation
    ---------------------
    poly_sat_h, poly_sat_degrees, poly_sat_time_method,
    errs_c_poly_sat, errs_grad_poly_sat
    """

    # Mesh sizes and time steps -----------------------------------------------
    hs: Optional[np.ndarray] = None
    dt: Optional[np.ndarray] = None

    # Spatial convergence: keys are polynomial degrees ------------------------
    errs_c_space:    Optional[Dict] = None
    errs_grad_space: Optional[Dict] = None

    # Polynomial convergence --------------------------------------------------
    poly_conv_h:       Optional[float] = None
    poly_conv_degrees: Optional[List]  = None
    errs_c_poly:       Optional[List]  = None
    errs_grad_poly:    Optional[List]  = None

    # Temporal convergence: keys are BDF orders or ThetaMethod ----------------
    errs_c_bdf:      Optional[Dict] = None
    errs_grad_bdf:   Optional[Dict] = None
    errs_c_theta:    Optional[Dict] = None
    errs_grad_theta: Optional[Dict] = None

    # Space saturation --------------------------------------------------------
    space_sat_time_method: Optional[TimeMethod] = None
    space_sat_hs:          Optional[np.ndarray] = None
    space_sat_degrees:     Optional[List]       = None
    errs_c_space_sat:      Optional[Dict]       = None
    errs_grad_space_sat:   Optional[Dict]       = None

    # Polynomial saturation ---------------------------------------------------
    poly_sat_h:           Optional[float]      = None
    poly_sat_degrees:     Optional[List]       = None
    poly_sat_time_method: Optional[TimeMethod] = None
    errs_c_poly_sat:      Optional[Dict]       = None
    errs_grad_poly_sat:   Optional[Dict]       = None

    def __post_init__(self):
        """Validate dict key types for all populated error fields."""
        if self.errs_c_space is not None:
            _check_keys("errs_c_space", self.errs_c_space, PolyDegree)
        if self.errs_grad_space is not None:
            _check_keys("errs_grad_space", self.errs_grad_space, PolyDegree)

        if self.errs_c_bdf is not None:
            _check_keys("errs_c_bdf", self.errs_c_bdf, BdfOrder)
        if self.errs_grad_bdf is not None:
            _check_keys("errs_grad_bdf", self.errs_grad_bdf, BdfOrder)

        if self.errs_c_theta is not None:
            _check_keys("errs_c_theta", self.errs_c_theta, ThetaMethod)
        if self.errs_grad_theta is not None:
            _check_keys("errs_grad_theta", self.errs_grad_theta, ThetaMethod)

        if self.space_sat_time_method is not None:
            _check_type("space_sat_time_method", self.space_sat_time_method, TimeMethod)
        if self.errs_c_space_sat is not None:
            _check_keys("errs_c_space_sat", self.errs_c_space_sat, PolyDegree)
            _check_nested_keys("errs_c_space_sat", self.errs_c_space_sat,
                               self.space_sat_time_method)
        if self.errs_grad_space_sat is not None:
            _check_keys("errs_grad_space_sat", self.errs_grad_space_sat, PolyDegree)
            _check_nested_keys("errs_grad_space_sat", self.errs_grad_space_sat,
                               self.space_sat_time_method)

        if self.poly_sat_time_method is not None:
            _check_type("poly_sat_time_method", self.poly_sat_time_method, TimeMethod)
        if self.errs_c_poly_sat is not None:
            _check_nested_time_keys("errs_c_poly_sat", self.errs_c_poly_sat,
                                    self.poly_sat_time_method)
        if self.errs_grad_poly_sat is not None:
            _check_nested_time_keys("errs_grad_poly_sat", self.errs_grad_poly_sat,
                                    self.poly_sat_time_method)


# Validation helpers ______________________________________________________________

def _check_type(field: str, value: object, expected_type: type) -> None:
    """Raise TypeError if value is not an instance of expected_type."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"ConvergenceData.{field}: expected {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _check_keys(field: str, d: dict, expected_key_type: type) -> None:
    """Raise TypeError if any key of d is not an instance of expected_key_type."""
    bad = [k for k in d if not isinstance(k, expected_key_type)]
    if bad:
        raise TypeError(
            f"ConvergenceData.{field}: all keys must be {expected_key_type.__name__}, "
            f"found {[type(k).__name__ for k in bad]}"
        )


def _check_nested_keys(field: str, d: dict, time_method: Optional[TimeMethod]) -> None:
    """Validate inner dict keys of a {PolyDegree: {BdfOrder|ThetaMethod: ...}} structure.

    The expected inner key type is inferred from time_method; if time_method is
    None the inner key type cannot be determined and validation is skipped.
    """
    if time_method is None:
        return
    inner_type = BdfOrder if time_method is TimeMethod.BDF else ThetaMethod
    for outer_key, inner in d.items():
        if not isinstance(inner, dict):
            raise TypeError(
                f"ConvergenceData.{field}[{outer_key}]: expected dict, "
                f"got {type(inner).__name__}"
            )
        bad = [k for k in inner if not isinstance(k, inner_type)]
        if bad:
            raise TypeError(
                f"ConvergenceData.{field}[{outer_key}]: all inner keys must be "
                f"{inner_type.__name__}, found {[type(k).__name__ for k in bad]}"
            )


def _check_nested_time_keys(field: str, d: dict,
                             time_method: Optional[TimeMethod]) -> None:
    """Validate keys of a {BdfOrder|ThetaMethod: ...} structure.

    The expected key type is inferred from time_method; if time_method is None
    validation is skipped.
    """
    if time_method is None:
        return
    key_type = BdfOrder if time_method is TimeMethod.BDF else ThetaMethod
    _check_keys(field, d, key_type)