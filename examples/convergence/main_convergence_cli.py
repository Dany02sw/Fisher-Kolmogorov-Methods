"""
Convergence study entry point — CLI.

Usage:
    python main_convergence_cli.py --solver dg_bdf --test cosine --conv spatial
    python main_convergence_cli.py --solver ldg_bdf --test wave --conv temporal \\
        --C11 1.0 --C12 0.5 --tol 1e-11 --max-it 200

Run ``python main_convergence_cli.py --help`` for the full option list.
"""

import argparse

from fisher_kolmogorov.models.solver_bdf         import SolverBDF
from fisher_kolmogorov.runners                   import (
    run_spatial_convergence,
    run_polynomial_convergence,
    run_temporal_convergence,
)
from fisher_kolmogorov.utilities.enum_utilities  import ConvType, TestType, PenaltyType
from fisher_kolmogorov.utilities.print_utilities import print_title, print_subtitle

from fisher_kolmogorov.configs.test_configs.cosine import (
    COSINE_SPATIAL_BDF,    COSINE_SPATIAL_THETA,
    COSINE_POLYNOMIAL_BDF, COSINE_POLYNOMIAL_THETA,
    COSINE_TEMPORAL_BDF,   COSINE_TEMPORAL_THETA,
)
from fisher_kolmogorov.configs.test_configs.wave import (
    WAVE_SPATIAL_BDF,    WAVE_SPATIAL_THETA,
    WAVE_POLYNOMIAL_BDF, WAVE_POLYNOMIAL_THETA,
    WAVE_TEMPORAL_BDF,   WAVE_TEMPORAL_THETA,
)

# Registry maps ___________________________________________________________________________________________________________________________
SOLVER_REGISTRY = {
    "dg_bdf"     : "fisher_kolmogorov.models.solver_dg_bdf",
    "dg_theta"   : "fisher_kolmogorov.models.solver_dg_theta",
    "ldg_bdf"    : "fisher_kolmogorov.models.solver_ldg_bdf",
    "ldg_theta"  : "fisher_kolmogorov.models.solver_ldg_theta",
    "ppdg_bdf"   : "fisher_kolmogorov.models.solver_ppdg_bdf",
    "ppdg_theta" : "fisher_kolmogorov.models.solver_ppdg_theta",
    "spldg_bdf"  : "fisher_kolmogorov.models.solver_spldg_bdf",
    "spldg_theta": "fisher_kolmogorov.models.solver_spldg_theta",
}

CLASS_NAME = {
    "dg_bdf"     : "SolverDgBDF",
    "dg_theta"   : "SolverDgTheta",
    "ldg_bdf"    : "SolverLdgBDF",
    "ldg_theta"  : "SolverLdgTheta",
    "ppdg_bdf"   : "SolverPpDgBDF",
    "ppdg_theta" : "SolverPpDgTheta",
    "spldg_bdf"  : "SolverSpLdgBDF",
    "spldg_theta": "SolverSpLdgTheta",
}

# (is_bdf, conv_type, test_type) -> TestConfig
CONFIG_REGISTRY = {
    (True,  ConvType.SPATIAL,    TestType.COSINE): COSINE_SPATIAL_BDF,
    (False, ConvType.SPATIAL,    TestType.COSINE): COSINE_SPATIAL_THETA,
    (True,  ConvType.POLYNOMIAL, TestType.COSINE): COSINE_POLYNOMIAL_BDF,
    (False, ConvType.POLYNOMIAL, TestType.COSINE): COSINE_POLYNOMIAL_THETA,
    (True,  ConvType.TEMPORAL,   TestType.COSINE): COSINE_TEMPORAL_BDF,
    (False, ConvType.TEMPORAL,   TestType.COSINE): COSINE_TEMPORAL_THETA,
    (True,  ConvType.SPATIAL,    TestType.WAVE):   WAVE_SPATIAL_BDF,
    (False, ConvType.SPATIAL,    TestType.WAVE):   WAVE_SPATIAL_THETA,
    (True,  ConvType.POLYNOMIAL, TestType.WAVE):   WAVE_POLYNOMIAL_BDF,
    (False, ConvType.POLYNOMIAL, TestType.WAVE):   WAVE_POLYNOMIAL_THETA,
    (True,  ConvType.TEMPORAL,   TestType.WAVE):   WAVE_TEMPORAL_BDF,
    (False, ConvType.TEMPORAL,   TestType.WAVE):   WAVE_TEMPORAL_THETA,
}

RUNNER_MAP = {
    ConvType.SPATIAL:    (run_spatial_convergence,    "Space convergence"),
    ConvType.POLYNOMIAL: (run_polynomial_convergence, "Polynomial degree convergence"),
    ConvType.TEMPORAL:   (run_temporal_convergence,   "Time convergence"),
}

TEST_TYPE_MAP = {t.name.lower(): t for t in TestType}
CONV_TYPE_MAP = {c.name.lower(): c for c in ConvType}


# Argument parsing ________________________________________________________________________________________________________________________
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run convergence studies for Fisher-Kolmogorov DG-based solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver", required=True, choices=SOLVER_REGISTRY.keys(),
                        help="Solver (snake_case module name, e.g. dg_bdf).")
    parser.add_argument("--test",   required=True, choices=[t.name.lower() for t in TestType],
                        help="Physical test case.")
    parser.add_argument("--conv",   required=True, choices=[c.name.lower() for c in ConvType],
                        help="Type of convergence study.")

    # Generic solver tolerances
    parser.add_argument("--tol",    type=float, default=1e-11, help="Nonlinear solver tolerance.")
    parser.add_argument("--max-it", type=int,   default=200,   help="Max nonlinear solver iterations.")
    parser.add_argument("--save-plot", action="store_true",    help="Save convergence plots to disk.")

    # Model-specific parameters (optional; ignored when not relevant to the chosen solver)
    parser.add_argument("--eta0",      type=float, help="DG / SP-LDG / PP-DG penalty coefficient.")
    parser.add_argument("--C11",       type=float, help="LDG stabilisation coefficient.")
    parser.add_argument("--C12",       type=float, help="LDG normal stabilisation coefficient.")
    parser.add_argument("--eps",       type=float, help="SP-LDG / PP-DG stabilisation coefficient.")
    parser.add_argument("--gamma",     type=float, help="DG penalty type (SIP=1, IIP=0, NIP=-1).")
    parser.add_argument("--theta",     type=float, help="SP-LDG averaging exponent.")
    parser.add_argument("--smoothing", type=float, help="PP-DG smoothing parameter.")
    parser.add_argument("--l",         type=int,   help="Polynomial degree (overrides config default).")

    return parser


def build_model_params(solver_key: str, args: argparse.Namespace):
    """Instantiate the correct ModelParams subclass from CLI args."""
    from fisher_kolmogorov.configs.model_configs import DgParams, LdgParams, SpLdgParams, PpDgParams

    DEFAULTS = {
        "dg_bdf"     : DgParams,
        "dg_theta"   : DgParams,
        "ldg_bdf"    : LdgParams,
        "ldg_theta"  : LdgParams,
        "ppdg_bdf"   : PpDgParams,
        "ppdg_theta" : PpDgParams,
        "spldg_bdf"  : SpLdgParams,
        "spldg_theta": SpLdgParams,
    }
    inst = DEFAULTS[solver_key]()

    overrides = {
        "eta_0"    : args.eta0,
        "C11"      : args.C11,
        "C12"      : args.C12,
        "eps"      : args.eps,
        "gamma"    : PenaltyType(args.gamma) if args.gamma is not None else None,
        "theta"    : args.theta,
        "smoothing": args.smoothing,
    }
    for field, value in overrides.items():
        if value is not None and hasattr(inst, field):
            setattr(inst, field, value)

    return inst


# Entry point _____________________________________________________________________________________________________________________________
if __name__ == "__main__":
    import importlib

    args      = build_parser().parse_args()
    conv_type = CONV_TYPE_MAP[args.conv]
    test_type = TEST_TYPE_MAP[args.test]

    # Load solver class
    module       = importlib.import_module(SOLVER_REGISTRY[args.solver])
    solver_class = getattr(module, CLASS_NAME[args.solver])

    # Select config from (is_bdf, conv_type, test_type)
    is_bdf = issubclass(solver_class, SolverBDF)
    config = CONFIG_REGISTRY[(is_bdf, conv_type, test_type)]

    # Build model params
    model_params = build_model_params(args.solver, args)

    runner, subtitle = RUNNER_MAP[conv_type]
    print_title(f"{CLASS_NAME[args.solver]}  ·  {config.name}")
    print_subtitle(subtitle)
    runner(
        solver_class=solver_class,
        config=config,
        model_params=model_params,
        tol=args.tol,
        max_it=args.max_it,
        save_plot=args.save_plot,
    )