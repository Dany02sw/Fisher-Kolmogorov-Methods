"""
Brain simulation entry point — CLI.

Usage:
    python main_brain_cli.py --solver ldg_theta --section sagittal
    python main_brain_cli.py --solver dg_bdf --section coronal \\
        --l 2 --T 50.0 --dt 0.25 --scheme bdf2 --eta0 10.0

Run ``python main_brain_cli.py --help`` for the full option list.
"""

import argparse
import importlib

from fisher_kolmogorov.utilities.enum_utilities  import (
    PolyDegree, BdfOrder, ThetaMethod, BrainSection, PenaltyType
)
from fisher_kolmogorov.utilities.print_utilities import print_title, print_subtitle

# Registry maps ___________________________________________________________________________________________________________________________
SOLVER_REGISTRY = {
    "dg_bdf"     : ("fisher_kolmogorov.models.solver_dg_bdf",     "SolverDgBDF"),
    "dg_theta"   : ("fisher_kolmogorov.models.solver_dg_theta",   "SolverDgTheta"),
    "ldg_bdf"    : ("fisher_kolmogorov.models.solver_ldg_bdf",    "SolverLdgBDF"),
    "ldg_theta"  : ("fisher_kolmogorov.models.solver_ldg_theta",  "SolverLdgTheta"),
    "ppdg_bdf"   : ("fisher_kolmogorov.models.solver_ppdg_bdf",   "SolverPpDgBDF"),
    "ppdg_theta" : ("fisher_kolmogorov.models.solver_ppdg_theta", "SolverPpDgTheta"),
    "spldg_bdf"  : ("fisher_kolmogorov.models.solver_spldg_bdf",  "SolverSpLdgBDF"),
    "spldg_theta": ("fisher_kolmogorov.models.solver_spldg_theta","SolverSpLdgTheta"),
}

SECTION_MAP = {s.name.lower(): s for s in BrainSection}

# Scheme strings → BdfOrder or ThetaMethod
SCHEME_MAP = {
    **{f"bdf{o.value}": o for o in BdfOrder},
    "ie": ThetaMethod.IE,
    "cn": ThetaMethod.CN,
    "ee": ThetaMethod.EE,
}


# Argument parsing ________________________________________________________________________________________________________________________
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a forward brain simulation for Fisher-Kolmogorov.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver",  required=True, choices=SOLVER_REGISTRY.keys(),
                        help="Solver (snake_case, e.g. ldg_theta).")
    parser.add_argument("--section", required=True, choices=[s.name.lower() for s in BrainSection],
                        help="Brain section to simulate.")

    # Simulation parameters (all optional — defaults come from SimulationParams dataclass)
    parser.add_argument("--l",      type=int,   default=PolyDegree.P2.value,
                        choices=[p.value for p in PolyDegree],
                        help="Polynomial degree.")
    parser.add_argument("--T",      type=float, default=50.0,  help="Final simulation time.")
    parser.add_argument("--dt",     type=float, default=0.25,  help="Time step size.")
    parser.add_argument("--scheme", type=str,   default="cn",  choices=SCHEME_MAP.keys(),
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
    parser.add_argument("--smoothing", type=float, help="PP-DG smoothing parameter.")

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
    from dolfin import *
    from fisher_kolmogorov.configs.test_configs.base     import BrainConfig, SimulationParams
    from fisher_kolmogorov.meshes.mesh_import            import mesh_factory
    from fisher_kolmogorov.utilities.enum_utilities      import MeshType
    from fisher_kolmogorov.utilities.initial_conditions  import get_initial_condition
    from fisher_kolmogorov.utilities.profiling_utilities import timer
    from examples.brain.main_brain                       import build_brain_pde_data

    args    = build_parser().parse_args()
    section = SECTION_MAP[args.section]

    brain_config = BrainConfig(plane=section)
    sim_params   = SimulationParams(
        l         = PolyDegree(args.l),
        T         = args.T,
        dt        = args.dt,
        nu_or_tht = SCHEME_MAP[args.scheme],
        tol       = args.tol,
        max_it    = args.max_it,
    )

    mod_path, cls_name = SOLVER_REGISTRY[args.solver]
    solver_class       = getattr(importlib.import_module(mod_path), cls_name)
    model_params       = build_model_params(args.solver, args)

    print_title("α-synuclein spreading — brain simulation")

    mesh, _, alpha, D, c_0 = build_brain_pde_data(brain_config)

    parameters["form_compiler"]["quadrature_degree"] = sim_params.l ** 2 + 4

    print_subtitle(f"Spreading on {mesh.name()}  ·  {cls_name}")

    solver = solver_class(mesh, D, alpha, c_0, **model_params.to_kwargs())
    with timer(f"Spreading on {mesh.name()}"):
        solver.Solve(
            t0    = sim_params.t0,
            dt    = sim_params.dt,
            T     = sim_params.T,
            order = sim_params.nu_or_tht,
            l     = sim_params.l,
            tol   = sim_params.tol,
            maxIt = sim_params.max_it,
        )