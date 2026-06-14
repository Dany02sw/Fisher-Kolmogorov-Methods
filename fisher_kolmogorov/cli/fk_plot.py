"""
fk-plot — plot convergence and/or saturation results for a Fisher-Kolmogorov solver.

Usage
-----
    fk-plot --space SPLDG --time BDF --plot TEMPORAL
    fk-plot --space SPLDG --time BDF --plot SPATIAL --save
    fk-plot --space LDG   --time BDF --study SATURATION --plot POLYNOMIAL
    fk-plot --space SPLDG --time BDF --plot POLYNOMIAL --combined
    fk-plot --config path/to/my_plot/

######################### NO MORE ###################################
In the first four forms the numerical data is loaded from the built-in
``models_config`` registry. In the fifth form the config folder must contain
an ``fk_config.py`` file that defines a ``plot_spec`` attribute (a
``PlotSpec`` instance) for fully custom layouts, or the same set of flags
used by the standard mode (``SPACE``, ``TIME``, ``PLOT``, ``STUDY``,
``COMBINED``, ``SAVE``, ``SpaceSatOrder``).

Standard flags
--------------
    --space     : space method (SPLDG, PPDG, LDG, DG)
    --time      : time  method (BDF, THETA)
    --plot      : convergence type to display (SPATIAL, POLYNOMIAL, TEMPORAL)
    --study     : study type (CONVERGENCE, SATURATION) — default CONVERGENCE
    --combined  : side-by-side polynomial + temporal plot (CONVERGENCE/POLYNOMIAL only)
    --save      : save the figure to the convergence results directory
    --sat-order : polynomial degree for single-degree spatial saturation (e.g. P2).
                  If omitted, spatial saturation defaults to the combined all-degrees
                  plot. Ignored for POLYNOMIAL saturation and all CONVERGENCE studies.

Custom config
-------------
    fk-plot --config path/to/my_plot/

    The folder must contain ``fk_config.py`` defining either:
      - ``plot_spec : PlotSpec`` for arbitrary subplot layouts, or
      - the same scalar flags (SPACE, TIME, PLOT, …) as in standard mode.
      In the scalar-flags path, ``SpaceSatOrder`` plays the same role as
      ``--sat-order``: if absent, spatial saturation defaults to combined.

Run ``fk-plot --help`` for the full option list.
"""

import argparse
import importlib.util
import os
import sys

from pathlib import Path

from fisher_kolmogorov.utilities.enum_utilities import (
    SpaceMethod, TimeMethod, ConvType, StudyType, PolyDegree,
)


# Model config registry __________________________________________________________________________________________________________________________
# DA CAMBIARE: O SI IMPONE DI LASCIARE LA CONFIGURAZIONE DENTRO IL FOLDER DI CONFIGURAZIONE, O SI FA PASSARE DA FUORI COME ARGOMENTO CLI
_PLOT_REGISTRY = {
    (SpaceMethod.SPLDG, TimeMethod.BDF):   "examples.plots.models_config.spldg_bdf",
    (SpaceMethod.SPLDG, TimeMethod.THETA): "examples.plots.models_config.spldg_theta",
    (SpaceMethod.PPDG,  TimeMethod.BDF):   "examples.plots.models_config.ppdg_bdf",
    (SpaceMethod.PPDG,  TimeMethod.THETA): "examples.plots.models_config.ppdg_theta",
    (SpaceMethod.LDG,   TimeMethod.BDF):   "examples.plots.models_config.ldg_bdf",
    (SpaceMethod.LDG,   TimeMethod.THETA): "examples.plots.models_config.ldg_theta",
    (SpaceMethod.DG,    TimeMethod.BDF):   "examples.plots.models_config.dg_bdf",
    (SpaceMethod.DG,    TimeMethod.THETA): "examples.plots.models_config.dg_theta",
}


# Argument parsing _______________________________________________________________________________________________________________________________

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fk-plot",
        description="Plot convergence and saturation results for a Fisher-Kolmogorov solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a config folder containing fk_config.py (custom mode).",
    )

    # Standard mode flags
    parser.add_argument("--space",  type=str, default=None, help="Space method (SPLDG, PPDG, LDG, DG).")
    parser.add_argument("--time",   type=str, default=None, help="Time method (BDF, THETA).")
    parser.add_argument("--plot",   type=str, default=None, help="Plot type (SPATIAL, POLYNOMIAL, TEMPORAL).")
    parser.add_argument("--study",  type=str, default="CONVERGENCE", help="Study type (CONVERGENCE, SATURATION).")
    parser.add_argument("--sat-order", type=str, default=None,
                        help="Polynomial degree for single-degree spatial saturation (e.g. P2). "
                             "If omitted, spatial saturation defaults to the combined all-degrees plot.")
    parser.add_argument("--combined", action="store_true",
                        help="Side-by-side polynomial + temporal plot (CONVERGENCE only).")
    parser.add_argument("--save",   action="store_true", help="Save the figure to disk.")

    return parser.parse_args(argv)


# Config loading _________________________________________________________________________________________________________________________________
def _load_config_module(config_dir: Path):
    """Dynamically import fk_config.py from config_dir."""
    config_file = config_dir / "fk_config.py"
    if not config_file.exists():
        print(f"[ERROR]: fk_config.py not found in {config_dir}")
        sys.exit(1)

    os.chdir(config_dir)

    spec   = importlib.util.spec_from_file_location("fk_config", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plot_data(space: SpaceMethod, time: TimeMethod):
    """Import the models_config module for the given (space, time) pair."""
    key = (space, time)
    if key not in _PLOT_REGISTRY:
        print(f"[ERROR]: No data available for ({space.name}, {time.name}).")
        sys.exit(1)
    return importlib.import_module(_PLOT_REGISTRY[key])


# Enum resolution ________________________________________________________________________________________________________________________________
def _resolve_enum(enum_cls, name, label):
    """Resolve a string to an enum member, exiting with a clear error on failure."""
    try:
        return enum_cls[name.upper()]
    except KeyError:
        valid = [m.name for m in enum_cls]
        print(f"[ERROR]: Invalid {label} '{name}'. Available: {valid}")
        sys.exit(1)


# Standard dispatch ______________________________________________________________________________________________________________________________

def _run_standard(args):
    """Execute the standard (non-custom) plot mode from CLI flags."""
    for flag, label in [("space", "--space"), ("time", "--time"), ("plot", "--plot")]:
        if getattr(args, flag) is None:
            print(f"[ERROR]: {label} is required in standard mode (no --config given).")
            sys.exit(1)

    space     = _resolve_enum(SpaceMethod, args.space, "--space")
    time      = _resolve_enum(TimeMethod,  args.time,  "--time")
    plot      = _resolve_enum(ConvType,    args.plot,  "--plot")
    study     = _resolve_enum(StudyType,   args.study, "--study")
    sat_order = _resolve_enum(PolyDegree,  args.sat_order, "--sat-order") \
                if args.sat_order is not None else None

    data = _load_plot_data(space, time)
    _dispatch(data, space, time, plot, study, sat_order, args.combined, args.save)


def _run_custom(config_dir: Path):
    """Execute custom mode: import fk_config.py and dispatch from its attributes."""
    module = _load_config_module(config_dir)

    # PlotSpec path
    plot_spec = getattr(module, "plot_spec", None)
    if plot_spec is not None:
        from fisher_kolmogorov.plots.plot_custom import plot_custom
        plot_custom(plot_spec)
        return

    # Scalar-flags path (same interface as standard mode)
    for attr in ("SPACE", "TIME", "PLOT"):
        if not hasattr(module, attr):
            print(f"[ERROR]: '{attr}' must be defined in fk_config.py (or provide 'plot_spec').")
            sys.exit(1)

    space     = module.SPACE
    time      = module.TIME
    plot      = module.PLOT
    study     = getattr(module, "STUDY",        StudyType.CONVERGENCE)
    sat_order = getattr(module, "SpaceSatOrder", PolyDegree.P2)
    combined  = getattr(module, "COMBINED",      False)
    save      = getattr(module, "SAVE",          False)

    data = _load_plot_data(space, time)
    _dispatch(data, space, time, plot, study, sat_order, combined, save)


# Core dispatch (shared by standard and scalar-flags custom) _____________________________________________________________________________________
def _dispatch(data, space, time, plot, study, sat_order, combined, save):
    """Route to the correct plot_utilities function based on study/plot flags."""
    from fisher_kolmogorov.plots.plot_utilities import (
        plot_spatial_convergence_all,
        plot_time_convergence_all,
        plot_polynomial_convergence,
        plot_spatial_saturation,
        plot_polynomial_saturation,
        plot_combined_poly_and_time,
        plot_spatial_saturation_combined,
    )

    if study == StudyType.CONVERGENCE:

        if plot == ConvType.SPATIAL:
            if time == TimeMethod.THETA:
                print("[ERROR]: Spatial convergence data not available for THETA method.")
                sys.exit(1)
            plot_spatial_convergence_all(
                data.hs, data.errs_c_space, data.errs_grad_space,
                space_method=space, time_method=time, save=save,
            )

        elif plot == ConvType.POLYNOMIAL:
            if time == TimeMethod.THETA:
                print("[ERROR]: Polynomial convergence data not available for THETA method.")
                sys.exit(1)
            if combined:
                plot_combined_poly_and_time(
                    errors_c_poly    = data.errs_c_poly,
                    errors_grad_poly = data.errs_grad_poly,
                    l_list           = data.PolyConvLList,
                    h                = data.PolyConvH,
                    dt_list          = data.dt,
                    errs_c_time      = data.errs_c_bdf,
                    time_method      = time,
                    space_method     = space,
                    save             = save,
                )
            else:
                plot_polynomial_convergence(
                    data.errs_c_poly, data.errs_grad_poly,
                    data.PolyConvLList, data.PolyConvH,
                    space_method=space, time_method=time, save=save,
                )

        elif plot == ConvType.TEMPORAL:
            if time == TimeMethod.BDF:
                plot_time_convergence_all(
                    data.dt, data.errs_c_bdf, data.errs_grad_bdf,
                    time_method=TimeMethod.BDF, space_method=space, save=save,
                )
            elif time == TimeMethod.THETA:
                plot_time_convergence_all(
                    data.dt, data.errs_c_theta, data.errs_grad_theta,
                    time_method=TimeMethod.THETA, space_method=space, save=save,
                )

        else:
            print(f"[ERROR]: Unknown plot type '{plot}'.")
            sys.exit(1)

    elif study == StudyType.SATURATION:

        if plot == ConvType.SPATIAL:
            if time == TimeMethod.THETA:
                print("[ERROR]: Spatial saturation data not available for THETA method.")
                sys.exit(1)
            # Single-degree only when --sat-order is given explicitly; otherwise combined
            if sat_order is not None:
                plot_spatial_saturation(
                    data.SpaceSatHs,
                    data.errs_c_space_sat_by_degree[sat_order],
                    data.errs_grad_space_sat_by_degree[sat_order],
                    l            = sat_order,
                    time_method  = data.SpaceSatTimeMethod,
                    space_method = space,
                    save         = save,
                )
            else:
                plot_spatial_saturation_combined(
                    data.SpaceSatHs,
                    data.errs_c_space_sat_by_degree,
                    data.errs_grad_space_sat_by_degree,
                    degrees      = data.SpaceSatDegrees,
                    time_method  = data.SpaceSatTimeMethod,
                    space_method = space,
                    save         = save,
                )

        elif plot == ConvType.POLYNOMIAL:
            if time == TimeMethod.THETA:
                print("[ERROR]: Polynomial saturation data not available for THETA method.")
                sys.exit(1)
            plot_polynomial_saturation(
                data.PolySatLList,
                data.errs_c_poly_sat,
                data.errs_grad_poly_sat,
                h            = data.PolySatH,
                time_method  = data.PolySatTimeMethod,
                space_method = space,
                save         = save,
            )

        elif plot == ConvType.TEMPORAL:
            print(f"[ERROR]: Saturation study not supported for TEMPORAL. "
                  f"Choose SPATIAL or POLYNOMIAL.")
            sys.exit(1)

        else:
            print(f"[ERROR]: Unknown plot type '{plot}'.")
            sys.exit(1)

    else:
        print(f"[ERROR]: Unknown study type '{study}'.")
        sys.exit(1)


# Entry point ____________________________________________________________________________________________________________________________________
def main(argv=None):
    args = _parse_args(argv)

    if args.config is not None:
        config_dir = args.config.resolve()
        if not config_dir.is_dir():
            print(f"[ERROR]: Config path is not a directory: {config_dir}")
            sys.exit(1)
        _run_custom(config_dir)
    else:
        _run_standard(args)


if __name__ == "__main__":
    main()
