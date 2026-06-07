"""
Wave · spatial convergence — saturation study (all l × all BDFν).

Outer loop: polynomial degree l = 1 … MAX_L.
Inner loop: BDF order ν = 1 … MAX_NU.

This ordering ensures that if the run is interrupted, at least the first l
degrees are complete with all BDF orders, allowing partial plots.

Already-completed (l, ν) pairs are skipped via checkpoint files in reproduce/.done/.

To rerun a single (l, ν) pair without touching this script:
    python3 examples/reproduce/py/wave/spatial/_wave_spatial_l2_bdf3.py

Usage:
    python3 examples/reproduce/py/wave/wave_spatial_saturation_all.py
"""

from fisher_kolmogorov.models.solver_spldg_bdf_red2 import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs        import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities     import ConvType, TestType

from examples.reproduce._run_reproduce import launch_reproduce, clear_done

MAX_L  = 3
MAX_NU = 6

if __name__ == "__main__":
    for l in range(2, MAX_L + 1):
        for nu in range(1, MAX_NU + 1):
            launch_reproduce(
                solver_class = SolverSpLdgBDFReduced2,
                model_params = SpLdgParams(eta_0=1.0, theta=-1.0),
                conv_type    = ConvType.SPATIAL,
                test_type    = TestType.WAVE,
                l            = l,
                nu           = nu,
                factory_kwargs = {"N_list": [10, 20, 35, 55, 70, 100]},
            )
    clear_done(ConvType.SPATIAL, TestType.WAVE)