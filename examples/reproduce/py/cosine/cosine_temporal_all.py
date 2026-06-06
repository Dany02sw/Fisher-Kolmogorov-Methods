"""
Cosine · temporal convergence — all BDF orders.

Runs BDF1 … BDF_MAX_NU in sequence. Each run requires a fine enough spatial
mesh to avoid pollution, so individual runs can be long.
Already-completed orders are skipped via checkpoint files in reproduce/.done/.

To rerun a single BDF order without touching this script:
    python3 examples/reproduce/py/cosine/temporal/_cosine_temporal_bdf<nu>.py

Usage:
    python3 examples/reproduce/py/cosine/cosine_temporal_all.py

"""
from fisher_kolmogorov.models.solver_spldg_bdf_red2 import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs        import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities     import ConvType, TestType
from fisher_kolmogorov.configs.test_configs.cosine  import make_c_exact_temporal_scaled

from examples.reproduce._run_reproduce import launch_reproduce, clear_done

MAX_NU = 6


if __name__ == "__main__":
    for nu in range(1, MAX_NU + 1):
        launch_reproduce(
            solver_class = SolverSpLdgBDFReduced2,
            model_params = SpLdgParams(eta_0=1.0, theta=-1.0),
            conv_type    = ConvType.TEMPORAL,
            test_type    = TestType.COSINE,
            nu           = nu,
            config_kwargs = {"c_exact": make_c_exact_temporal_scaled(0.75)},
        )
    clear_done(ConvType.TEMPORAL, TestType.COSINE)
