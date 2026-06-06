"""
Wave · polynomial convergence — single BDF order fallback: BDF3.

Run this script directly when wave_polynomial_all.py was interrupted
and only this specific order needs to be (re)launched.

Usage:
    python3 examples/reproduce/py/wave/polynomial/_wave_polynomial_bdf3.py
"""

from fisher_kolmogorov.models.solver_spldg_bdf_red2 import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs        import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities     import ConvType, TestType

from reproduce._run_reproduce import launch_reproduce

_NU = 3

if __name__ == "__main__":
    launch_reproduce(
        solver_class = SolverSpLdgBDFReduced2,
        model_params = SpLdgParams(eta_0=1.0, theta=-1.0),
        conv_type    = ConvType.POLYNOMIAL,
        test_type    = TestType.WAVE,
        nu           = _NU,
    )
