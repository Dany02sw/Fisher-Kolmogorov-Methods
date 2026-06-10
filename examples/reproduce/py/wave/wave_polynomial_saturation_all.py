"""
Wave · polynomial convergence — saturation study (all BDFν).

Runs BDF1 … MAX_NU in sequence over the full polynomial degree range.
Already-completed orders are skipped via checkpoint files in reproduce/.done/.

To rerun a single BDF order without touching this script:
    python3 examples/reproduce/py/wave/polynomial/_wave_polynomial_bdf6.py

Usage:
    python3 examples/reproduce/py/wave/wave_polynomial_saturation_all.py
"""

from fisher_kolmogorov.models.solver_factory    import make_solver_class
from fisher_kolmogorov.configs.model_configs    import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType, PolyDegree, SpaceMethod, TimeMethod

from examples.reproduce._run_reproduce import launch_reproduce, clear_done

MAX_NU = 6

if __name__ == "__main__":
    for nu in range(1, MAX_NU + 1):
        launch_reproduce(
            solver_class   = make_solver_class(
                space  = SpaceMethod.SPLDG,
                time   = TimeMethod.BDF,
                params = SpLdgParams(eta_0=1.0, theta=-1.0, smoothing=1e-12),
                full   = False,
            ),
            conv_type      = ConvType.POLYNOMIAL,
            test_type      = TestType.WAVE,
            nu             = nu,
            tol            = 1e-10,
            max_it         = 300,
            factory_kwargs = {
                "N_fixed": 16,
                "l_list" : [PolyDegree(i) for i in range(1, 7)],
            },
        )
    clear_done(ConvType.POLYNOMIAL, TestType.WAVE)
