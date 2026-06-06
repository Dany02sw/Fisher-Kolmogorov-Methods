"""
Cosine · temporal convergence — single BDF order fallback: BDF6.

Run this script directly when cosine_temporal_all.py was interrupted
and only this specific order needs to be (re)launched.

Usage:
    python3 examples/reproduce/py/cosine/temporal/_cosine_temporal_bdf6.py
"""

from fisher_kolmogorov.models.solver_spldg_bdf_red2 import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs        import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities     import ConvType, TestType
from fisher_kolmogorov.configs.test_configs.cosine  import make_c_exact_temporal_scaled

from reproduce._run_reproduce import launch_reproduce

_NU = 6

if __name__ == "__main__":
    launch_reproduce(
        solver_class  = SolverSpLdgBDFReduced2,
        model_params  = SpLdgParams(eta_0=1.0, theta=-1.0),
        conv_type     = ConvType.TEMPORAL,
        test_type     = TestType.COSINE,
        nu            = _NU,
        config_kwargs = {"c_exact": make_c_exact_temporal_scaled(0.75)},
    )
