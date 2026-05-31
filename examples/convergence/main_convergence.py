"""
Convergence study entry point — Python configuration.

Edit the four variables in the "CONFIGURATION" block and run with:

    python main_convergence.py
"""

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

from fisher_kolmogorov.models.solver_dg_bdf    import SolverDgBDF      # swap to any of the 8 solver classes
from fisher_kolmogorov.configs.model_configs    import DgParams          # matching ModelParams subclass
from fisher_kolmogorov.configs.test_configs    import COSINE_SPATIAL    # swap to any TestConfig instance
from fisher_kolmogorov.utilities.enum_utilities import ConvType

SOLVER_CLASS = SolverDgBDF
MODEL_PARAMS = DgParams(eta_0=10.0)
CONFIG       = COSINE_SPATIAL
CONV_TYPE    = ConvType.SPATIAL              # SPATIAL | POLYNOMIAL | TEMPORAL

# Solver tolerances (override per-run if needed)
TOL    = 1e-11
MAX_IT = 200

# ── END CONFIGURATION ─────────────────────────────────────────────────────────

from fisher_kolmogorov.runners import run_spatial_convergence, run_polynomial_convergence, run_temporal_convergence
from fisher_kolmogorov.utilities.print_utilities import print_title, print_subtitle

if __name__ == "__main__":
    print_title(f"{SOLVER_CLASS.__name__}  ·  {CONFIG.name}")

    runner_map = {
        ConvType.SPATIAL    : (run_spatial_convergence,    "Space convergence test"),
        ConvType.POLYNOMIAL : (run_polynomial_convergence, "Polynomial degree convergence test"),
        ConvType.TEMPORAL   : (run_temporal_convergence,   "Time convergence test"),
    }

    runner, subtitle = runner_map[CONV_TYPE]
    print_subtitle(subtitle)
    runner(
        solver_class=SOLVER_CLASS,
        config=CONFIG,
        model_params=MODEL_PARAMS,
        tol=TOL,
        max_it=MAX_IT,
    )
