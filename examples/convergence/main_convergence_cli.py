"""
Convergence study entry point — CLI.

Usage:
    python main_convergence_cli.py --model DgBDF --test cosine_spatial --conv spatial
    python main_convergence_cli.py --model LdgBDF --test cosine_temporal --conv temporal \\
        --C11 10.0 --C12 0.5 --tol 1e-11 --max-it 200

Run ``python main_convergence_cli.py --help`` for the full option list.
"""

import argparse
import importlib

from fisher_kolmogorov.runners import run_spatial_convergence, run_polynomial_convergence, run_temporal_convergence
from fisher_kolmogorov.utilities.enum_utilities  import PenaltyType
from fisher_kolmogorov.utilities.print_utilities import print_title, print_subtitle

# Registry maps ___________________________________________________________________________________________________________________________
SOLVER_REGISTRY = {
    "DgBDF"     : ("SolverDgBDF",      "SolverDgBDF"),
    "DgTheta"   : ("SolverDgTheta",    "SolverDgTheta"),
    "LdgBDF"    : ("SolverLdgBDF",     "SolverLdgBDF"),
    "LdgTheta"  : ("SolverLdgTheta",   "SolverLdgTheta"),
    "PpDgBDF"   : ("SolverPpDgBDF",    "SolverPpDgBDF"),
    "PpDgTheta" : ("SolverPpDgTheta",  "SolverPpDgTheta"),
    "SpLdgBDF"  : ("SolverSpLdgBDF",   "SolverSpLdgBDF"),
    "SpLdgTheta": ("SolverSpLdgTheta", "SolverSpLdgTheta"),
}

CONFIG_REGISTRY = {
    "cosine_spatial"   : ("TestConfigs.cosine", "COSINE_SPATIAL"),
    "cosine_temporal"  : ("TestConfigs.cosine", "COSINE_TEMPORAL"),
    "wave_spatial"     : ("TestConfigs.wave",   "WAVE_SPATIAL"),
    # "my_new_test" : ("TestConfigs.my_module", "MY_CONFIG"),
}

RUNNER_MAP = {
    "spatial"    : (run_spatial_convergence,    "Space convergence test"),
    "polynomial" : (run_polynomial_convergence, "Polynomial degree convergence test"),
    "temporal"   : (run_temporal_convergence,   "Time convergence test"),
}


# Argument parsing ________________________________________________________________________________________________________________________
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run convergence studies for DG based solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",  required=True, choices=SOLVER_REGISTRY.keys(),
                        help="Solver class to use.")
    parser.add_argument("--test",   required=True, choices=CONFIG_REGISTRY.keys(),
                        help="Test configuration (problem data + mesh settings).")
    parser.add_argument("--conv",   required=True, choices=RUNNER_MAP.keys(),
                        help="Type of convergence study to run.")

    # Generic solver tolerances
    parser.add_argument("--tol",    type=float, default=1e-11, help="Nonlinear solver tolerance.")
    parser.add_argument("--max-it", type=int,   default=200,   help="Max nonlinear solver iterations.")
    parser.add_argument("--save-plot", action="store_true",    help="Save convergence plots to disk.")

    # Model-specific parameters (all optional; ignored when not relevant)
    parser.add_argument("--eta_0",     type=float, help="DG / SP-LDG / PP-DG penalty coefficient.")
    parser.add_argument("--C11",       type=float, help="LDG stabilisation coefficient.")
    parser.add_argument("--C12",       type=float, help="LDG normal stabilisation coefficient.")
    parser.add_argument("--eps",       type=float, help="SP-LDG / PP-DG stabilisation coefficient.")
    parser.add_argument("--gamma",     type=float, help="DG penalty type (SIP=1, IIP=0, NIP=-1).")
    parser.add_argument("--theta",     type=float, help="SP-LDG averaging exponent.")
    parser.add_argument("--smoothing", type=float, help="Transformations smoothing parameter.")

    return parser


def build_model_params(model_key: str, args: argparse.Namespace):
    """Instantiate the correct ModelParams subclass from CLI args."""
    from fisher_kolmogorov.configs.model_configs import DgParams, LdgParams, SpLdgParams, PpDgParams

    defaults = {
        "DgBDF"     : DgParams,
        "DgTheta"   : DgParams,
        "LdgBDF"    : LdgParams,
        "LdgTheta"  : LdgParams,
        "PpDgBDF"   : PpDgParams,
        "PpDgTheta" : PpDgParams,
        "SpLdgBDF"  : SpLdgParams,
        "SpLdgTheta": SpLdgParams,
    }
    cls  = defaults[model_key]
    inst = cls()  # start from dataclass defaults

    # Override only fields the user explicitly provided
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
    args = build_parser().parse_args()

    # Load solver class dynamically
    mod_name, cls_name = SOLVER_REGISTRY[args.model]
    solver_class = getattr(importlib.import_module(mod_name), cls_name)

    # Load test config
    mod_name, obj_name = CONFIG_REGISTRY[args.test]
    config = getattr(importlib.import_module(mod_name), obj_name)

    # Build model params
    model_params = build_model_params(args.model, args)

    # Run
    runner, subtitle = RUNNER_MAP[args.conv]
    print_title(f"{args.model}  ·  {config.name}")
    print_subtitle(subtitle)
    runner(
        solver_class=solver_class,
        config=config,
        model_params=model_params,
        tol=args.tol,
        max_it=args.max_it,
        save_plot=args.save_plot,
    )
