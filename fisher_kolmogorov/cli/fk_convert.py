"""
fk-convert — convert a Gmsh .msh file to a FEniCS-compatible .xdmf file.

Usage
-----
    fk-convert --input path/to/mesh.msh
    fk-convert --input path/to/mesh.msh --output-dir fk-xdmf/
"""

import argparse
import sys

from pathlib import Path

from fisher_kolmogorov.meshes.brain_meshes.converter.gmsh_to_fenics import msh_to_xdmf


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fk-convert",
        description="Convert a Gmsh .msh file to a FEniCS-compatible .xdmf file (MSH → XDMF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True, type=Path,
                        help="Path to the input .msh file.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for the output .xdmf and .h5 files. "
                             "Defaults to fk-xdmf/ in the current working directory.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    input_file   = args.input
    output_dir   = args.output_dir if args.output_dir is not None else Path.cwd() / "fk-xdmf"

    if not input_file.exists():
        print(f"[ERROR]: File not found: {input_file}")
        sys.exit(1)

    msh_to_xdmf(input_file, output_dir)


if __name__ == "__main__":
    main()
