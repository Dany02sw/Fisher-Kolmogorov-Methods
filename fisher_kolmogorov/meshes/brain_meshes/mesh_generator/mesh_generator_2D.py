from __future__ import annotations

import trimesh
import numpy as np
import gmsh
import sys

from pathlib import Path


# ===== Mesh configuration =====
MESH_CONFIG = {
    "lc"               : 1.0,
    "simplify_tol"     : 0.1,
    "size_min"         : 0.8,
    "size_max"         : 1.2,
    "optimize"         : True,
    "optimize_netgen"  : True,
    "scale_factor"     : 1.0,
}

# ===== Section definitions =====
SECTIONS = {
    "sagittal"   : {"normal": [1, 0, 0], "default_offset":  20.0, "axis": "x"},
    "coronal"    : {"normal": [0, 1, 0], "default_offset":   0.0, "axis": "y"},
    "horizontal" : {"normal": [0, 0, 1], "default_offset":  10.0, "axis": "z"},
}



# ===== Mesh generator =====
def generate_brain_mesh(stl_path: Path, output_path: Path, section: str, offset: float,
                        normal: list):
    """
    Generate a 2-D brain mesh from a .stl file and save it as .msh.

    Parameters
    ----------
    stl_path    : Path  — path to the input .stl file
    output_path : Path  — path to the output .msh file
    section     : str   — label used for display only (e.g. 'sagittal' or 'custom')
    offset      : float — offset along the section normal axis
    normal      : list  — plane normal as a 3-element list, e.g. [1, 0, 0]
    """
    cfg = MESH_CONFIG
    lc  = cfg["lc"]

    print(f"\nLoading STL: {stl_path.name}...")
    mesh = trimesh.load(str(stl_path))
    if not mesh.is_watertight:
        print("[WARNING]: Mesh is not watertight. Results may be unreliable.")

    print(f"Slicing along {section} plane (normal={normal}, offset={offset:+.2f})...")
    origin   = mesh.centroid + np.array(normal) * offset
    slice_3d = mesh.section(plane_normal=normal, plane_origin=origin)
    if slice_3d is None:
        print("[ERROR]: The cutting plane does not intersect the mesh. Try a different offset.")
        sys.exit(1)

    slice_2d, _ = slice_3d.to_planar()
    print(f"Number of contours: {len(slice_2d.polygons_full)}")
    slice_2d.show()

    all_polygons = slice_2d.polygons_full[:-1]

    gmsh.initialize()
    gmsh.model.add(f"brain_{section}")

    gmsh.option.setNumber("Mesh.MeshSizeMin",    cfg["size_min"])
    gmsh.option.setNumber("Mesh.MeshSizeMax",    cfg["size_max"])
    gmsh.option.setNumber("Mesh.Optimize",       int(cfg["optimize"]))
    gmsh.option.setNumber("Mesh.OptimizeNetgen", int(cfg["optimize_netgen"]))

    surfaces              = []
    white_matter_surfaces = []
    physical_boundaries   = {"exterior": [], "holes": [], "white_matter": []}
    cl_poly_4             = None

    for i, poly in enumerate(all_polygons):
        poly_sim = poly.simplify(tolerance=cfg["simplify_tol"], preserve_topology=True)
        coords   = np.array(poly_sim.exterior.coords) * cfg["scale_factor"]

        poly_point_tags = []
        for c in coords[:-1]:
            p_tag = gmsh.model.geo.addPoint(c[0], c[1], 0, lc)
            poly_point_tags.append(p_tag)

        l_tag_ext = gmsh.model.geo.addBSpline(poly_point_tags + [poly_point_tags[0]])
        cl_ext    = gmsh.model.geo.addCurveLoop([l_tag_ext])

        if i == 4:
            cl_poly_4 = cl_ext

        current_surface_loops = [cl_ext]
        physical_boundaries["exterior"].append(l_tag_ext)

        wm_tag = None
        for j, interior in enumerate(poly_sim.interiors):
            coords_int = np.array(interior.coords) * cfg["scale_factor"]
            p_tags_int = []
            for c in coords_int[:-1]:
                p_tag = gmsh.model.geo.addPoint(c[0], c[1], 0, lc)
                p_tags_int.append(p_tag)

            l_tag_int = gmsh.model.geo.addBSpline(p_tags_int + [p_tags_int[0]])
            cl_int    = gmsh.model.geo.addCurveLoop([l_tag_int])

            wm_loops = [cl_int]
            if cl_poly_4 is not None and wm_tag is not None:
                wm_loops.append(wm_tag)

            wm_tag = gmsh.model.geo.addPlaneSurface([cl_int])
            white_matter_surfaces.append(wm_tag)

            current_surface_loops.append(cl_int)
            physical_boundaries["white_matter"].append(l_tag_int)

        s_tag = gmsh.model.geo.addPlaneSurface(current_surface_loops)
        surfaces.append(s_tag)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, surfaces, tag=1)
    gmsh.model.setPhysicalName(2, 1, "gray_matter")

    if white_matter_surfaces:
        gmsh.model.addPhysicalGroup(2, white_matter_surfaces, tag=2)
        gmsh.model.setPhysicalName(2, 2, "white_matter")

    gmsh.model.addPhysicalGroup(1, physical_boundaries["exterior"], tag=10)
    gmsh.model.setPhysicalName(1, 10, "exterior_boundary")

    if physical_boundaries["holes"]:
        gmsh.model.addPhysicalGroup(1, physical_boundaries["holes"], tag=11)
        gmsh.model.setPhysicalName(1, 11, "internal_boundaries")

    gmsh.model.mesh.generate(2)

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))
    gmsh.finalize()

    print(f"Mesh successfully generated: {output_path.name}")


# ===== Interactive logic =====
def _select_stl(stl_dir: Path) -> Path:
    """Prompt the user to choose one of the .stl files found in stl_dir."""
    stl_files = sorted(stl_dir.glob("*.stl"))
    if not stl_files:
        print(f"\n[ERROR]: No .stl files found in: {stl_dir.resolve()}")
        print(f"Place your .stl file there and try again.")
        print(f"If you do not have a brain .stl model, refer to the download link in the README.")
        sys.exit(1)

    print("\nAvailable .stl files:")
    for i, f in enumerate(stl_files):
        print(f"  [{i}] {f.name}")

    try:
        choice = int(input("\nInsert the number corresponding to the .stl file to use: "))
        if choice < 0 or choice >= len(stl_files):
            raise ValueError
    except ValueError:
        print("[ERROR]: Invalid choice.")
        sys.exit(1)

    return stl_files[choice]


def _select_section() -> tuple[str, list, str]:
    """
    Prompt the user to choose a section or provide a custom normal.

    Returns
    -------
    section_name : str   — label for display and default filename
    normal       : list  — 3-element plane normal
    axis_label   : str   — axis label for offset prompt ('x', 'y', 'z', or 'offset')
    """
    section_names = list(SECTIONS.keys())

    print("\nAvailable sections:")
    for i, name in enumerate(section_names):
        sec = SECTIONS[name]
        print(f"  [{i}] {name:<12} (normal={sec['normal']},  default offset {sec['axis']}={sec['default_offset']:+.1f})")
    print(f"  [{len(section_names)}] custom       (specify normal manually)")

    try:
        choice = int(input("\nInsert the number corresponding to the section: "))
        if choice < 0 or choice > len(section_names):
            raise ValueError
    except ValueError:
        print("[ERROR]: Invalid choice.")
        sys.exit(1)

    if choice < len(section_names):
        name = section_names[choice]
        return name, SECTIONS[name]["normal"], SECTIONS[name]["axis"]

    # Custom normal
    print("\nEnter the three components of the plane normal (space-separated, e.g. '1 0 0'):")
    try:
        parts  = input("  normal [nx ny nz]: ").strip().split()
        normal = [float(p) for p in parts]
        if len(normal) != 3:
            raise ValueError
    except ValueError:
        print("[ERROR]: Invalid normal. Provide exactly three floats.")
        sys.exit(1)

    norm = np.linalg.norm(normal)
    if norm == 0:
        print("[ERROR]: Normal vector cannot be zero.")
        sys.exit(1)
    normal = [v / norm for v in normal]
    print(f"  Normalized normal: {[f'{v:.4f}' for v in normal]}")

    return "custom", normal, "offset"


def _select_offset(section_name: str, axis_label: str) -> float:
    """Prompt the user for the offset, using the section default when available."""
    default = SECTIONS[section_name]["default_offset"] if section_name in SECTIONS else 0.0
    try:
        raw    = input(f"Insert offset along {axis_label} axis [default {default:+.1f}]: ").strip()
        return float(raw) if raw else default
    except ValueError:
        print("[ERROR]: Invalid offset value.")
        sys.exit(1)


def _select_output_name(section_name: str, output_dir: Path) -> Path:
    """
    Prompt the user for the output filename, defaulting to the section name.

    Returns the full output Path (including directory and .msh extension).
    """
    raw = input(f"Output filename (without extension) [{section_name}]: ").strip()
    name = raw if raw else section_name
    return output_dir / f"{name}.msh"


def main():
    """Interactive logic for brain mesh generation."""
    script_dir       = Path(__file__).parent.resolve()
    stl_dir          = script_dir / ".." / "stl_files"
    output_directory = script_dir / ".." / "msh_files"

    print("\n")
    print(40 * "#")
    print("### Brain mesh generator: STL -> MSH ###")
    print(40 * "#")

    # --- Ensure stl_dir exists ---
    if not stl_dir.exists():
        stl_dir.mkdir(parents=True)
        print(f"\n[WARNING]: The folder 'stl_files' did not exist and has been created at:")
        print(f"  {stl_dir.resolve()}")
        print(f"Please place your .stl file there and run the script again.")
        print(f"If you do not have a brain .stl model, refer to the download link in the README.")
        sys.exit(0)

    stl_path                     = _select_stl(stl_dir)
    section_name, normal, axis   = _select_section()
    offset                       = _select_offset(section_name, axis)
    output_path                  = _select_output_name(section_name, output_directory)

    generate_brain_mesh(
        stl_path    = stl_path,
        output_path = output_path,
        section     = section_name,
        offset      = offset,
        normal      = normal,
    )


if __name__ == "__main__":
    main()