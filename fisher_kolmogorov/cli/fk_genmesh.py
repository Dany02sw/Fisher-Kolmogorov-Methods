"""
fk-genmesh — generate a 2-D brain mesh from a .stl file.

Usage
-----
    fk-genmesh --stl path/to/brain.stl --output fk-xdmf/brain.msh \\
               --section sagittal --offset 20.0

All meshing parameters have sensible defaults; most users only need
--stl and --section.
"""

import argparse
import sys

from pathlib import Path

from fisher_kolmogorov.meshes.brain_meshes.mesh_generator.mesh_generator_2d import generate_brain_mesh


# Predefined section normals based on the default stl model _______________________________________________________________________________________________________
_SECTION_NORMALS = {
    "sagittal"  : [1.0, 0.0, 0.0],
    "coronal"   : [0.0, 1.0, 0.0],
    "horizontal": [0.0, 0.0, 1.0],
}

_DEFAULT_OFFSETS = {
    "sagittal"  : 20.0,
    "coronal"   :  0.0,
    "horizontal": 10.0,
}


# Parser __________________________________________________________________________________________________________________________________________________________
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fk-genmesh",
        description="Generate a 2-D brain mesh from a .stl file (STL → MSH).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--stl",     required=True, type=Path, help="Path to the input .stl file.")
    parser.add_argument("--section", required=True, choices=list(_SECTION_NORMALS) + ["custom"],
                        help="Anatomical section to slice, or 'custom' to provide --normal manually.")

    # Output
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to the output .msh file. Defaults to fk-xdmf/<section>.msh.")

    # Slice parameters
    parser.add_argument("--offset", type=float, default=None,
                        help="Offset along the plane normal from the mesh centroid. "
                             "Defaults to the standard value for the chosen section.")
    parser.add_argument("--normal", type=float, nargs=3, metavar=("NX", "NY", "NZ"),
                        help="Custom plane normal (required when --section custom).")

    # Meshing parameters
    parser.add_argument("--lc",              type=float, default=1.0,  help="Characteristic mesh size.")
    parser.add_argument("--simplify-tol",    type=float, default=0.1,  help="Polygon simplification tolerance.")
    parser.add_argument("--size-min",        type=float, default=0.8,  help="Minimum element size.")
    parser.add_argument("--size-max",        type=float, default=1.2,  help="Maximum element size.")
    parser.add_argument("--scale-factor",    type=float, default=1.0,  help="Scale factor for polygon coordinates.")
    parser.add_argument("--no-optimize",     action="store_true",       help="Disable the default Gmsh optimizer.")
    parser.add_argument("--no-optimize-netgen", action="store_true",    help="Disable the Netgen optimizer.")
    parser.add_argument("--show-gui",        action="store_true",       help="Open the Gmsh GUI after generation.")

    return parser.parse_args(argv)


# Main and entry point ___________________________________________________________________________________________________________________________________________
def main(argv=None):
    args = _parse_args(argv)

    # Resolve normal and offset
    if args.section == "custom":
        if args.normal is None:
            print("[ERROR]: --normal NX NY NZ is required when --section custom.")
            sys.exit(1)
        normal  = args.normal
        offset  = args.offset if args.offset is not None else 0.0
        section = "custom"
    else:
        normal  = _SECTION_NORMALS[args.section]
        offset  = args.offset if args.offset is not None else _DEFAULT_OFFSETS[args.section]
        section = args.section

    # Resolve output path
    output_path = args.output
    if output_path is None:
        out_dir     = Path.cwd() / "fk-xdmf"
        output_path = out_dir / f"{section}.msh"

    generate_brain_mesh(
        stl_path        = args.stl,
        output_path     = output_path,
        section         = section,
        offset          = offset,
        normal          = normal,
        lc              = args.lc,
        simplify_tol    = args.simplify_tol,
        size_min        = args.size_min,
        size_max        = args.size_max,
        scale_factor    = args.scale_factor,
        optimize        = not args.no_optimize,
        optimize_netgen = not args.no_optimize_netgen,
        show_gui        = args.show_gui,
    )


if __name__ == "__main__":
    main()
