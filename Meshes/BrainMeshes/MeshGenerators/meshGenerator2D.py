import trimesh
import numpy as np
import gmsh
import sys

from pathlib import Path


# ===== Mesh configuration =====
# Edit these values to change the mesh quality without touching the interactive logic.

MESH_CONFIG = {
    "lc"               : 1.0,   # characteristic length (used for gmsh points)
    "simplify_tol"     : 0.1,   # polygon simplification tolerance
    "size_min"         : 0.8,   # Mesh.MeshSizeMin
    "size_max"         : 1.2,   # Mesh.MeshSizeMax
    "optimize"         : True,  # Mesh.Optimize
    "optimize_netgen"  : True,  # Mesh.OptimizeNetgen
    "scale_factor"     : 1.0,   # coordinate scale factor
}

# ===== Section definitions =====
# Each entry: (label, plane_normal, offset_axis_index)
# offset_axis_index: 0=x, 1=y, 2=z

SECTIONS = {
    "sagittal"   : {"normal": [1, 0, 0], "default_offset":  20.0, "axis": "x"},
    "coronal"    : {"normal": [0, 1, 0], "default_offset":   0.0, "axis": "y"},
    "horizontal" : {"normal": [0, 0, 1], "default_offset":  10.0, "axis": "z"},
}


# ===== Mesh generator =====
def generate_brain_mesh(stl_path: Path, output_path: Path, section: str, offset: float):
    """
    Generate a 2-D brain mesh from a .stl file and save it as .msh.

    Parameters
    ----------
    stl_path    : Path   — path to the input .stl file
    output_path : Path   — path to the output .msh file
    section     : str    — anatomical section: 'sagittal', 'coronal', or 'horizontal'
    offset      : float  — offset along the section normal axis
    """
    cfg     = MESH_CONFIG
    sec     = SECTIONS[section]
    normal  = sec["normal"]
    axis    = sec["axis"]
    lc      = cfg["lc"]

    # --- Load STL ---
    print(f"\nLoading STL: {stl_path.name}...")
    mesh = trimesh.load(str(stl_path))
    if not mesh.is_watertight:
        print("[WARNING]: Mesh is not watertight. Results may be unreliable.")

    # --- Slice ---
    offset_array          = np.array([0.0, 0.0, 0.0])
    axis_index            = {"x": 0, "y": 1, "z": 2}[axis]
    offset_array[axis_index] = offset
    origin                = mesh.centroid + offset_array

    print(f"Slicing along {section} plane (normal={normal}, offset {axis}={offset:+.2f})...")
    slice_3d = mesh.section(plane_normal=normal, plane_origin=origin)
    if slice_3d is None:
        print("[ERROR]: The cutting plane does not intersect the mesh. Try a different offset.")
        sys.exit(1)

    slice_2d, _ = slice_3d.to_planar()
    print(f"Number of contours: {len(slice_2d.polygons_full)}")
    slice_2d.show()

    # --- Build gmsh geometry ---
    all_polygons = slice_2d.polygons_full[:-1]

    gmsh.initialize()
    gmsh.model.add(f"brain_{section}")

    gmsh.option.setNumber("Mesh.MeshSizeMin",      cfg["size_min"])
    gmsh.option.setNumber("Mesh.MeshSizeMax",      cfg["size_max"])
    gmsh.option.setNumber("Mesh.Optimize",         int(cfg["optimize"]))
    gmsh.option.setNumber("Mesh.OptimizeNetgen",   int(cfg["optimize_netgen"]))

    surfaces              = []
    white_matter_surfaces = []
    physical_boundaries   = {"exterior": [], "holes": [], "white_matter": []}
    cl_poly_4             = None

    for i, poly in enumerate(all_polygons):
        poly_sim = poly.simplify(tolerance=cfg["simplify_tol"], preserve_topology=True)
        coords   = np.array(poly_sim.exterior.coords) * cfg["scale_factor"]

        # Exterior points and curve
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

        # Interior (white matter) holes
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

    # Physical groups
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

    # Generate and save
    gmsh.model.mesh.generate(2)

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))
    gmsh.finalize()

    print(f"Mesh successfully generated: {output_path.name}")


# ===== Interactive logic =====
def main():
    """
    Interactive logic for brain mesh generation.
    """
    script_dir       = Path(__file__).parent.resolve()
    stl_path         = script_dir / ".." / "StlFiles" / "brain.stl"
    output_directory = script_dir / ".." / "MshFiles"

    print("\n")
    print(36*"#")
    print("### Brain mesh generator: STL -> MSH ###")
    print(37*"#")

    # --- Check STL exists ---
    if not stl_path.exists():
        print(f"\n[FATAL ERROR]: STL file not found at: {stl_path.resolve()}")
        print("Place 'brain.stl' inside the 'StlFiles' folder and try again.")
        sys.exit(1)

    # --- Choose section ---
    section_names = list(SECTIONS.keys())
    print("\nAvailable sections:")
    for i, name in enumerate(section_names):
        sec = SECTIONS[name]
        print(f"  [{i}] {name:<12} (normal={sec['normal']},  default offset {sec['axis']}={sec['default_offset']:+.1f})")

    try:
        choice = int(input("\nInsert the number corresponding to the section: "))
        if choice < 0 or choice >= len(section_names):
            raise ValueError
    except ValueError:
        print("[ERROR]: Invalid choice.")
        sys.exit(1)

    section = section_names[choice]
    sec     = SECTIONS[section]

    # --- Choose offset ---
    default_offset = sec["default_offset"]
    try:
        raw = input(f"Insert offset along {sec['axis']} axis [default {default_offset:+.1f}]: ").strip()
        offset = float(raw) if raw else default_offset
    except ValueError:
        print("[ERROR]: Invalid offset value.")
        sys.exit(1)

    # --- Output filename ---
    output_path = output_directory / f"{section}.msh"

    generate_brain_mesh(
        stl_path    = stl_path,
        output_path = output_path,
        section     = section,
        offset      = offset,
    )


# ===== Main =====
if __name__ == "__main__":
    main()