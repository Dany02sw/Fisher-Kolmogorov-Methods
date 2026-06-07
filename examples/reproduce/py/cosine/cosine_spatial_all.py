"""
Cosine · spatial convergence — all polynomial degrees.

Runs l = 1 … MAX_L in sequence. Already-completed degrees are skipped
automatically via checkpoint files in reproduce/.done/.

To rerun a single degree without touching this script:
    python3 examples/reproduce/py/cosine/spatial/_cosine_spatial_l<degree>.py

Usage:
    python3 examples/reproduce/py/cosine/cosine_spatial_all.py
"""

from fisher_kolmogorov.models.solver_spldg_bdf_red2 import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs        import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities     import ConvType, TestType

from examples.reproduce._run_reproduce import launch_reproduce, clear_done

MAX_L = 8

if __name__ == "__main__":
    for l in range(1, MAX_L + 1):
        launch_reproduce(
            solver_class   = SolverSpLdgBDFReduced2,
            model_params   = SpLdgParams(eta_0=1.0, theta=-1.0),
            conv_type      = ConvType.SPATIAL,
            test_type      = TestType.COSINE,
            l              = l,
            tol            = 1e-12,
            max_it         = 300,
            factory_kwargs = {"N_ref": [2, 3, 4, 5]},
        )
    clear_done(ConvType.SPATIAL, TestType.COSINE)
