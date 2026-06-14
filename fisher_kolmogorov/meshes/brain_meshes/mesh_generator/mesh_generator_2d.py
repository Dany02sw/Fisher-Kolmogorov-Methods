from __future__ import annotations

import trimesh
import numpy as np
import gmsh
import sys

from pathlib import Path


def generate_brain_mesh(
    stl_path         : Path,
    output_path      : Path,
    section          : str,
    offset           : float,
    normal           : list[float],
    lc               : float = 1.0,
    simplify_tol     : float = 0.1,
    size_min         : float = 0.8,
    size_max         : float = 1.2,
    scale_factor     : float = 1.0,
    optimize         : bool  = True,
    optimize_netgen  : bool  = True,
    show_gui         : bool  = False,
) -> None:
    """
    Generate a 2-D brain mesh from a .stl file and save it as .msh.

    Parameters
    ----------
    stl_path        : Path
        Path to the input .stl file.
    output_path     : Path
        Path to the output .msh file.
    section         : str
        Label used for display and the Gmsh model name (e.g. 'sagittal').
    offset          : float
        Offset along the plane normal from the mesh centroid.
    normal          : list of float
        Plane normal as a 3-element list, e.g. [1, 0, 0].
    lc              : float
        Characteristic mesh size for Gmsh point generation.
    simplify_tol    : float
        Tolerance for polygon simplification (Shapely).
    size_min        : float
        Minimum element size passed to Gmsh.
    size_max        : float
        Maximum element size passed to Gmsh.
    scale_factor    : float
        Scale factor applied to polygon coordinates before meshing.
    optimize        : bool
        Whether to run the default Gmsh mesh optimizer.
    optimize_netgen : bool
        Whether to run the Netgen optimizer after the default one.
    show_gui        : bool
        If True, open the Gmsh GUI after mesh generation.
    """
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

    gmsh.option.setNumber("Mesh.MeshSizeMin",    size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax",    size_max)
    gmsh.option.setNumber("Mesh.Optimize",       int(optimize))
    gmsh.option.setNumber("Mesh.OptimizeNetgen", int(optimize_netgen))

    surfaces              = []
    white_matter_surfaces = []
    physical_boundaries   = {"exterior": [], "holes": [], "white_matter": []}
    cl_poly_4             = None

    for i, poly in enumerate(all_polygons):
        poly_sim = poly.simplify(tolerance=simplify_tol, preserve_topology=True)
        coords   = np.array(poly_sim.exterior.coords) * scale_factor

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
            coords_int = np.array(interior.coords) * scale_factor
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

    if show_gui and "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))
    gmsh.finalize()

    print(f"Mesh successfully generated: {output_path.name}")
