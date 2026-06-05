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
 
from fisher_kolmogorov.models.solver_bdf import SolverBDF
from fisher_kolmogorov.runners           import (
    run_spatial_convergence,
    run_polynomial_convergence,
    run_temporal_convergence,
)
from fisher_kolmogorov.utilities.enum_utilities  import (
    ConvType, TestType, PenaltyType, MeshStructure, PolyDegree, BdfOrder, ThetaMethod,
)
from fisher_kolmogorov.utilities.print_utilities import print_title, print_subtitle
 
from fisher_kolmogorov.configs.test_configs.cosine import (
    make_cosine_spatial, make_cosine_polynomial, make_cosine_temporal,
    make_c_exact_temporal_scaled,
)
from fisher_kolmogorov.configs.test_configs.wave import (
    make_wave_spatial, make_wave_polynomial, make_wave_temporal,
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
 
# (is_bdf, conv_type, test_type) -> factory callable
FACTORY_REGISTRY = {
    (True,  ConvType.SPATIAL,    TestType.COSINE): make_cosine_spatial,
    (False, ConvType.SPATIAL,    TestType.COSINE): make_cosine_spatial,
    (True,  ConvType.POLYNOMIAL, TestType.COSINE): make_cosine_polynomial,
    (False, ConvType.POLYNOMIAL, TestType.COSINE): make_cosine_polynomial,
    (True,  ConvType.TEMPORAL,   TestType.COSINE): make_cosine_temporal,
    (False, ConvType.TEMPORAL,   TestType.COSINE): make_cosine_temporal,
    (True,  ConvType.SPATIAL,    TestType.WAVE):   make_wave_spatial,
    (False, ConvType.SPATIAL,    TestType.WAVE):   make_wave_spatial,
    (True,  ConvType.POLYNOMIAL, TestType.WAVE):   make_wave_polynomial,
    (False, ConvType.POLYNOMIAL, TestType.WAVE):   make_wave_polynomial,
    (True,  ConvType.TEMPORAL,   TestType.WAVE):   make_wave_temporal,
    (False, ConvType.TEMPORAL,   TestType.WAVE):   make_wave_temporal,
}
 
RUNNER_MAP = {
    ConvType.SPATIAL:    (run_spatial_convergence,    "Space convergence"),
    ConvType.POLYNOMIAL: (run_polynomial_convergence, "Polynomial degree convergence"),
    ConvType.TEMPORAL:   (run_temporal_convergence,   "Time convergence"),
}
 
# Scheme strings -> BdfOrder or ThetaMethod
SCHEME_MAP = {
    **{f"bdf{o.value}": o for o in BdfOrder},
    "ie": ThetaMethod.IE,
    "cn": ThetaMethod.CN,
    "ee": ThetaMethod.EE,
}
 
TEST_TYPE_MAP  = {t.name.lower(): t for t in TestType}
CONV_TYPE_MAP  = {c.name.lower(): c for c in ConvType}
STRUCT_MAP     = {s.name.lower(): s for s in MeshStructure}


# Argument parsing ________________________________________________________________________________________________________________________
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run convergence studies for Fisher-Kolmogorov DG-based solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
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

    # Mesh parameters
    parser.add_argument("--N-ref",  type=int,   nargs="+",
                        help="Mesh refinement exponents (mesh size = 2**n). "
                             "Ignored if --N-list is provided.")
    parser.add_argument("--N-list", type=int,   nargs="+",
                        help="Explicit mesh sizes, bypasses --N-ref construction.")
    parser.add_argument("--N-fixed", type=int,
                        help="Fixed mesh size for polynomial / temporal studies.")
    parser.add_argument("--mesh-structure", choices=STRUCT_MAP.keys(), default=None,
                        help="Mesh structure (structured | unstructured).")
 
    # Time discretization
    parser.add_argument("--scheme", type=str, default=None, choices=SCHEME_MAP.keys(),
                        help="Time integration scheme (bdf1..bdf6 | ie | cn | ee). "
                             "Overrides the factory default.")
    parser.add_argument("--dt-list", type=float, nargs="+",
                        help="Explicit list of time steps for temporal studies.")
 
    # Polynomial degree
    parser.add_argument("--l", type=int,
                        help="Polynomial degree for spatial / temporal studies "
                             "(overrides factory default).")
    parser.add_argument("--l-list", type=int, nargs="+",
                        help="List of polynomial degrees for polynomial studies "
                             "(overrides factory default).")
 
    # Exact solution scaling (temporal cosine, edge cases)
    parser.add_argument("--temporal-coeff", type=float, default=None,
                        help="Exponent coefficient for the temporal exact solution "
                             "c_space * exp(-coeff * t). Default is 1.0. "
                             "Use a smaller value if Newton does not converge "
                             "(known case: SPLDG with BDF2).")

    # Model-specific parameters (optional; ignored when not relevant to the chosen solver)
    parser.add_argument("--eta0",      type=float, help="DG / SP-LDG / PP-DG penalty coefficient.")
    parser.add_argument("--C11",       type=float, help="LDG stabilisation coefficient.")
    parser.add_argument("--C12",       type=float, help="LDG normal stabilisation coefficient.")
    parser.add_argument("--eps",       type=float, help="SP-LDG / PP-DG stabilisation coefficient.")
    parser.add_argument("--gamma",     type=float, help="DG penalty type (SIP=1, IIP=0, NIP=-1).")
    parser.add_argument("--theta",     type=float, help="SP-LDG averaging exponent.")
    parser.add_argument("--smoothing", type=float, help="PP-DG smoothing parameter.")

    # kwargs
    parser.add_argument(
        "--linearize", action="store_true", help="Enable linearization of the reaction term (dg_bdf / ldg_bdf only)."
    )

    return parser


# Helpers _________________________________________________________________________________________________________________________________
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


def build_solver_kwargs(solver_key: str, args: argparse.Namespace) -> dict:
    """Collect solver-specific keyword arguments from CLI args."""
    kwargs = {}
    if solver_key in ("dg_bdf", "ldg_bdf") and args.linearize:
        kwargs["Linearize"] = True
    return kwargs


def build_config(factory, args: argparse.Namespace, conv_type: ConvType):
    """
    Call the appropriate factory with parameters resolved from CLI args.
 
    Parameters that are None fall through to the factory defaults.
    The study-specific kwarg dict (spatial_kwargs, polynomial_kwargs,
    temporal_kwargs) is built dynamically from the conv_type.
    """
    study_key = {
        ConvType.SPATIAL:    "spatial_kwargs",
        ConvType.POLYNOMIAL: "polynomial_kwargs",
        ConvType.TEMPORAL:   "temporal_kwargs",
    }[conv_type]
 
    # Build study-level overrides — only include keys that were explicitly set
    study_overrides = {}
    if args.dt_list:
        study_overrides["dt_list"] = args.dt_list
 
    # Build config-level overrides
    config_overrides = {}
    if args.temporal_coeff is not None:
        config_overrides["c_exact"] = make_c_exact_temporal_scaled(args.temporal_coeff)
 
    # Resolve scheme
    nu_or_tht = SCHEME_MAP[args.scheme] if args.scheme else None
 
    # Resolve mesh structure
    mesh_structure = STRUCT_MAP[args.mesh_structure] if args.mesh_structure else None
 
    # Resolve polynomial degree(s)
    l_space = PolyDegree(args.l)                   if args.l      else None
    l_list  = [PolyDegree(v) for v in args.l_list] if args.l_list else None
 
    # Build factory kwargs — only pass what was explicitly provided
    factory_kwargs = {}
 
    if nu_or_tht      is not None: factory_kwargs["nu_or_tht"]      = nu_or_tht
    if mesh_structure is not None: factory_kwargs["mesh_structure"] = mesh_structure
 
    if conv_type == ConvType.SPATIAL:
        if l_space is not None:  factory_kwargs["l_space"] = l_space
        if args.N_ref:           factory_kwargs["N_ref"]   = args.N_ref
        if args.N_list:          factory_kwargs["N_list"]  = args.N_list
 
    elif conv_type == ConvType.POLYNOMIAL:
        if l_list is not None:       factory_kwargs["l_list"]  = l_list
        if args.N_fixed is not None: factory_kwargs["N_fixed"] = args.N_fixed
 
    elif conv_type == ConvType.TEMPORAL:
        if l_space is not None:      factory_kwargs["l_space"] = l_space
        if args.N_fixed is not None: factory_kwargs["N_fixed"] = args.N_fixed
 
    if study_overrides:
        factory_kwargs[study_key]  = study_overrides
    if config_overrides:
        factory_kwargs["config_kwargs"] = config_overrides
 
    return factory(**factory_kwargs)


# Entry point _____________________________________________________________________________________________________________________________
def main():
    args      = build_parser().parse_args()
    conv_type = CONV_TYPE_MAP[args.conv]
    test_type = TEST_TYPE_MAP[args.test]

    module       = importlib.import_module(SOLVER_REGISTRY[args.solver])
    solver_class = getattr(module, CLASS_NAME[args.solver])

    is_bdf  = issubclass(solver_class, SolverBDF)
    factory = FACTORY_REGISTRY[(is_bdf, conv_type, test_type)]
    config  = build_config(factory, args, conv_type)

    model_params  = build_model_params(args.solver, args)
    solver_kwargs = build_solver_kwargs(args.solver, args)

    runner, subtitle = RUNNER_MAP[conv_type]
    print_title(f"{CLASS_NAME[args.solver]}  ·  {config.name}")
    print_subtitle(subtitle)
    runner(
        solver_class  = solver_class,
        config        = config,
        model_params  = model_params,
        tol           = args.tol,
        max_it        = args.max_it,
        save_plot     = args.save_plot,
        **solver_kwargs,
    )


if __name__ == "__main__":
    main()

 