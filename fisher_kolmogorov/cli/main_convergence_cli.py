"""
Convergence study entry point — CLI.

Usage:
    fk-convergence --solver dg_bdf --test cosine --conv spatial
    fk-convergence --solver ldg_bdf --test wave --conv temporal \\
        --C11 1.0 --C12 0.5 --tol 1e-11 --max-it 200

Run ``fk-convergence --help`` for the full option list.
"""

import argparse
import importlib

from pathlib import Path

from fisher_kolmogorov.utilities.enum_utilities import (
    ConvType, TestType, PenaltyType, MeshStructure, PolyDegree, BdfOrder, ThetaMethod,
)
from fisher_kolmogorov.configs.test_configs.cosine import make_c_exact_temporal_scaled
from fisher_kolmogorov.cli._run_core               import launch

# Solver registry _________________________________________________________________________________________________________________________
_SOLVER_REGISTRY = {
    "dg_bdf"           : ("fisher_kolmogorov.models.solver_dg_bdf",           "SolverDgBDF"),
    "dg_theta"         : ("fisher_kolmogorov.models.solver_dg_theta",         "SolverDgTheta"),
    "ldg_bdf"          : ("fisher_kolmogorov.models.solver_ldg_bdf",          "SolverLdgBDF"),
    "ldg_theta"        : ("fisher_kolmogorov.models.solver_ldg_theta",        "SolverLdgTheta"),
    "ppdg_bdf"         : ("fisher_kolmogorov.models.solver_ppdg_bdf",         "SolverPpDgBDF"),
    "ppdg_theta"       : ("fisher_kolmogorov.models.solver_ppdg_theta",       "SolverPpDgTheta"),
    "spldg_bdf"        : ("fisher_kolmogorov.models.solver_spldg_bdf",        "SolverSpLdgBDF"),
    "spldg_theta"      : ("fisher_kolmogorov.models.solver_spldg_theta",      "SolverSpLdgTheta"),
    "spldg_bdf_red2"   : ("fisher_kolmogorov.models.solver_spldg_bdf_red2",   "SolverSpLdgBDFReduced2"),
    "spldg_theta_red2" : ("fisher_kolmogorov.models.solver_spldg_theta_red2", "SolverSpLdgThetaReduced2"),
}

_SCHEME_MAP    = {**{f"bdf{o.value}": o for o in BdfOrder}, "ie": ThetaMethod.IE, "cn": ThetaMethod.CN, "ee": ThetaMethod.EE}
_TEST_TYPE_MAP = {t.name.lower(): t for t in TestType}
_CONV_TYPE_MAP = {c.name.lower(): c for c in ConvType}
_STRUCT_MAP    = {s.name.lower(): s for s in MeshStructure}


# Argument parsing ________________________________________________________________________________________________________________________
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run convergence studies for Fisher-Kolmogorov DG-based solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--solver", required=True, choices=_SOLVER_REGISTRY.keys(),
                        help="Solver (snake_case, e.g. dg_bdf).")
    parser.add_argument("--test",   required=True, choices=[t.name.lower() for t in TestType],
                        help="Physical test case.")
    parser.add_argument("--conv",   required=True, choices=[c.name.lower() for c in ConvType],
                        help="Type of convergence study.")

    parser.add_argument("--tol",       type=float, default=1e-11, help="Nonlinear solver tolerance.")
    parser.add_argument("--max-it",    type=int,   default=200,   help="Max nonlinear solver iterations.")
    parser.add_argument("--save-plot",   action="store_true",     help="Save convergence plots to disk.")
    parser.add_argument("--output-dir",  type=str, default=None,   help="Directory for XDMF solution export.")

    parser.add_argument("--N-ref",  type=int, nargs="+",
                        help="Mesh refinement exponents (mesh size = 2**n). Ignored if --N-list is provided.")
    parser.add_argument("--N-list", type=int, nargs="+",
                        help="Explicit mesh sizes, bypasses --N-ref construction.")
    parser.add_argument("--N-fixed", type=int,
                        help="Fixed mesh size for polynomial / temporal studies.")
    parser.add_argument("--mesh-structure", choices=_STRUCT_MAP.keys(), default=None,
                        help="Mesh structure (structured | unstructured).")

    parser.add_argument("--scheme",  type=str, default=None, choices=_SCHEME_MAP.keys(),
                        help="Time integration scheme (bdf1..bdf6 | ie | cn | ee).")
    parser.add_argument("--dt-list", type=float, nargs="+",
                        help="Explicit list of time steps for temporal studies.")

    parser.add_argument("--l",      type=int,
                        help="Polynomial degree for spatial / temporal studies.")
    parser.add_argument("--l-list", type=int, nargs="+",
                        help="List of polynomial degrees for polynomial studies.")

    parser.add_argument("--temporal-coeff", type=float, default=None,
                        help="Exponent coefficient for c_space * exp(-coeff * t). "
                             "Use a smaller value if Newton does not converge (known case: SPLDG with BDF2).")

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
    from fisher_kolmogorov.configs.model_configs import DgParams, LdgParams, SpLdgParams, PpDgParams

    DEFAULTS = {
        "dg_bdf"         : DgParams,    "dg_theta"         : DgParams,
        "ldg_bdf"        : LdgParams,   "ldg_theta"        : LdgParams,
        "ppdg_bdf"       : PpDgParams,  "ppdg_theta"       : PpDgParams,
        "spldg_bdf"      : SpLdgParams, "spldg_theta"      : SpLdgParams,
        "spldg_bdf_red2" : SpLdgParams, "spldg_theta_red2" : SpLdgParams,
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


def _build_factory_kwargs(args: argparse.Namespace, conv_type: ConvType) -> dict:
    """
    Resolve CLI args into factory keyword arguments.
    Only explicitly provided args are included; everything else falls through
    to the factory defaults.
    """
    study_key = {
        ConvType.SPATIAL:    "spatial_kwargs",
        ConvType.POLYNOMIAL: "polynomial_kwargs",
        ConvType.TEMPORAL:   "temporal_kwargs",
    }[conv_type]

    study_overrides = {}
    if args.dt_list:
        study_overrides["dt_list"] = args.dt_list

    config_overrides = {}
    if args.temporal_coeff is not None:
        config_overrides["c_exact"] = make_c_exact_temporal_scaled(args.temporal_coeff)

    factory_kwargs = {}

    if args.scheme:         factory_kwargs["nu_or_tht"]      = _SCHEME_MAP[args.scheme]
    if args.mesh_structure: factory_kwargs["mesh_structure"] = _STRUCT_MAP[args.mesh_structure]

    if conv_type == ConvType.SPATIAL:
        if args.l:      factory_kwargs["l_space"] = PolyDegree(args.l)
        if args.N_ref:  factory_kwargs["N_ref"]   = args.N_ref
        if args.N_list: factory_kwargs["N_list"]  = args.N_list

    elif conv_type == ConvType.POLYNOMIAL:
        if args.l_list:              factory_kwargs["l_list"]  = [PolyDegree(v) for v in args.l_list]
        if args.N_fixed is not None: factory_kwargs["N_fixed"] = args.N_fixed

    elif conv_type == ConvType.TEMPORAL:
        if args.l:                   factory_kwargs["l_space"] = PolyDegree(args.l)
        if args.N_fixed is not None: factory_kwargs["N_fixed"] = args.N_fixed

    if study_overrides:  factory_kwargs[study_key]       = study_overrides
    if config_overrides: factory_kwargs["config_kwargs"] = config_overrides

    return factory_kwargs


# Entry point _____________________________________________________________________________________________________________________________
def main():
    args      = _build_parser().parse_args()
    conv_type = _CONV_TYPE_MAP[args.conv]
    test_type = _TEST_TYPE_MAP[args.test]

    mod_path, cls_name = _SOLVER_REGISTRY[args.solver]
    solver_class       = getattr(importlib.import_module(mod_path), cls_name)
    model_params       = _build_model_params(args.solver, args)

    solver_kwargs = {}
    if args.solver in ("dg_bdf", "ldg_bdf") and args.linearize:
        solver_kwargs["Linearize"] = True

    launch(
        solver_class   = solver_class,
        model_params   = model_params,
        conv_type      = conv_type,
        test_type      = test_type,
        tol            = args.tol,
        max_it         = args.max_it,
        factory_kwargs = _build_factory_kwargs(args, conv_type),
        output_dir     = Path(args.output_dir) if args.output_dir else None,
        save_plot      = args.save_plot,
        **solver_kwargs,
    )


if __name__ == "__main__":
    main()
