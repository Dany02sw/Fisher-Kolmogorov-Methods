"""
Wave · polynomial convergence — model comparison (all BDF solvers, l = 1, 2).

Runs a polynomial convergence study for each of the 4 BDF-based solvers
(DG, LDG, PP-DG, SP-LDG-Red2) on the travelling-wave test case with
polynomial degrees l = 1 and l = 2. Solutions are exported to XDMF for
visual comparison.

Usage:
    python examples/reproduce/py/wave/wave_polynomial_comparison.py
"""

from pathlib import Path

from fisher_kolmogorov.models.solver_dg_bdf          import SolverDgBDF
from fisher_kolmogorov.models.solver_ldg_bdf         import SolverLdgBDF
from fisher_kolmogorov.models.solver_ppdg_bdf        import SolverPpDgBDF
from fisher_kolmogorov.models.solver_spldg_bdf_red2  import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs         import DgParams, LdgParams, PpDgParams, SpLdgParams
from fisher_kolmogorov.utilities.enum_utilities      import ConvType, TestType, PolyDegree, BdfOrder, PenaltyType
from examples.reproduce._run_reproduce               import launch_reproduce, clear_done
from config import WAVE_DIR

_SOLVERS = [
    (SolverDgBDF,            DgParams(eta_0=10.0, gamma=PenaltyType.SIP)),
    (SolverLdgBDF,           LdgParams(C11=1.0, C12=0.5)),
    (SolverPpDgBDF,          PpDgParams(eta_0=10.0)),
    (SolverSpLdgBDFReduced2, SpLdgParams(eta_0=1.0, theta=-1.0, smoothing=1e-12)),
]

if __name__ == "__main__":
    for solver_class, model_params in _SOLVERS:
        output_dir = WAVE_DIR / solver_class.__name__
        output_dir.mkdir(parents=True, exist_ok=True)
        launch_reproduce(
            solver_class   = solver_class,
            model_params   = model_params,
            conv_type      = ConvType.POLYNOMIAL,
            test_type      = TestType.WAVE,
            study_name     = "comparison",
            output_dir     = output_dir,
            tol            = 1e-10,
            max_it         = 300,
            factory_kwargs = {"l_list": [PolyDegree.P1, PolyDegree.P2], "nu_or_tht": BdfOrder.BDF6},
        )
    clear_done(ConvType.POLYNOMIAL, TestType.WAVE, study_name="comparison")