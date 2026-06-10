"""
Brain simulation entry point — CLI.

Usage:
    fk-brain --solver spldg_bdf --section sagittal
    fk-brain --solver spldg_bdf --section coronal \\
        --l 2 --T 50.0 --dt 0.25 --scheme bdf2 --eta0 1.0

Run ``fk-brain --help`` for the full option list.
"""

import argparse

from fisher_kolmogorov.models.solver_factory     import make_solver_class
from fisher_kolmogorov.utilities.enum_utilities  import (
    TimeMethod, PolyDegree, BdfOrder, ThetaMethod, BrainSection, PenaltyType,
)
from fisher_kolmogorov.cli._registries import _SOLVER_REGISTRY, _PARAMS_DEFAULTS, _SCHEME_MAP

_SECTION_MAP = {s.name.lower(): s for s in BrainSection}


# Argument parsing ________________________________________________________________________________________________________________________
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a forward brain simulation for Fisher-Kolmogorov.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver",  required=True, choices=_SOLVER_REGISTRY.keys(),
                        help="Solver (snake_case, e.g. spldg_bdf).")
    parser.add_argument("--section", required=True, choices=[s.name.lower() for s in BrainSection],
                        help="Brain section to simulate.")

    # Simulation parameters (all optional — defaults come from SimulationParams dataclass)
    parser.add_argument("--l",      type=int,   default=PolyDegree.P2.value,
                        choices=[p.value for p in PolyDegree], help="Polynomial degree.")
    parser.add_argument("--T",      type=float, default=50.0,  help="Final simulation time.")
    parser.add_argument("--dt",     type=float, default=0.25,  help="Time step size.")
    parser.add_argument("--scheme", type=str,   default="bdf2", choices=_SCHEME_MAP.keys(),
                        help="Time integration scheme (bdf1..bdf6 | ie | cn | ee).")
    parser.add_argument("--tol",    type=float, default=1e-6,  help="Nonlinear solver tolerance.")
    parser.add_argument("--max-it", type=int,   default=500,   help="Max nonlinear solver iterations.")

    # Model-specific parameters (optional; ignored when not relevant)
    parser.add_argument("--eta0",      type=float, help="DG / SP-LDG / PP-DG penalty coefficient.")
    parser.add_argument("--C11",       type=float, help="LDG stabilisation coefficient.")
    parser.add_argument("--C12",       type=float, help="LDG normal stabilisation coefficient.")
    parser.add_argument("--eps",       type=float, help="SP-LDG / PP-DG stabilisation coefficient.")
    parser.add_argument("--gamma",     type=float, help="DG penalty type (SIP=1, IIP=0, NIP=-1).")
    parser.add_argument("--theta",     type=float, help="SP-LDG averaging exponent.")
    parser.add_argument("--smoothing", type=float, help="SP-LDG / PP-DG smoothing parameter.")
    parser.add_argument("--linearize", action="store_true",
                        help="Enable linearization of the reaction term (dg_bdf / ldg_bdf only).")

    return parser


# Helpers _________________________________________________________________________________________________________________________________
def _build_model_params(solver_key: str, args: argparse.Namespace):
    """Instantiate the correct ModelParams subclass from CLI args."""
    inst = _PARAMS_DEFAULTS[solver_key]()

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
def main():
    from fisher_kolmogorov.configs.test_configs.base  import SimulationParams
    from fisher_kolmogorov.configs.test_configs.brain import make_brain_config
    from fisher_kolmogorov.runners.brain              import run_brain_simulation

    args    = _build_parser().parse_args()
    section = _SECTION_MAP[args.section]

    sim_params = SimulationParams(
        l         = PolyDegree(args.l),
        T         = args.T,
        dt        = args.dt,
        nu_or_tht = _SCHEME_MAP[args.scheme],
        tol       = args.tol,
        max_it    = args.max_it,
    )

    space, time, full = _SOLVER_REGISTRY[args.solver]
    model_params      = _build_model_params(args.solver, args)
    linearize         = args.linearize and time is TimeMethod.BDF

    solver_class = make_solver_class(
        space     = space,
        time      = time,
        params    = model_params,
        linearize = linearize,
        full      = full,
    )

    run_brain_simulation(
        solver_class = solver_class,
        sim_config   = make_brain_config(section),
        sim_params   = sim_params,
    )


if __name__ == "__main__":
    main()
