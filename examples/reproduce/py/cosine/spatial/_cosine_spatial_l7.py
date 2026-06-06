"""
Cosine · spatial convergence — single degree fallback: l = 7.

Run this script directly when cosine_spatial_all.py was interrupted
and only this specific degree needs to be (re)launched.

Usage:
    python3 examples/reproduce/py/cosine/spatial/_cosine_spatial_l7.py
"""

from fisher_kolmogorov.models.solver_spldg_bdf_red2 import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs        import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities     import ConvType, TestType

from reproduce._run_reproduce import launch_reproduce

_L = 7

if __name__ == "__main__":
    launch_reproduce(
        solver_class = SolverSpLdgBDFReduced2,
        model_params = SpLdgParams(eta_0=1.0, theta=-1.0),
        conv_type    = ConvType.SPATIAL,
        test_type    = TestType.COSINE,
        l            = _L,
    )
