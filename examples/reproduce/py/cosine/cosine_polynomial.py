"""
Cosine · polynomial convergence.

Single run: one fixed BDFν over all polynomial degrees.
No checkpoint needed — the run is short enough to complete in one shot.

Usage:
    python3 examples/reproduce/py/cosine/cosine_polynomial.py
"""

from fisher_kolmogorov.models.solver_factory    import make_solver_class
from fisher_kolmogorov.configs.model_configs    import SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities import ConvType, TestType, PolyDegree, SpaceMethod, TimeMethod

from examples.reproduce._run_reproduce import launch_reproduce, clear_done


if __name__ == "__main__":
    launch_reproduce(
        solver_class   = make_solver_class(
            space  = SpaceMethod.SPLDG,
            time   = TimeMethod.BDF,
            params = SpLdgParams(eta_0=1.0, theta=-1.0),
            full   = False,
        ),
        conv_type      = ConvType.POLYNOMIAL,
        test_type      = TestType.COSINE,
        tol            = 1e-12,
        max_it         = 300,
        factory_kwargs = {
            "N_fixed": 5,
            "l_list" : list(PolyDegree),
        },
    )
    clear_done(ConvType.POLYNOMIAL, TestType.COSINE)
