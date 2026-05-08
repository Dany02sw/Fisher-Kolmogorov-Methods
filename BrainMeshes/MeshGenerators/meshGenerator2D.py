import trimesh
from pathlib import Path
import numpy as np
import gmsh
import sys
from pathlib import Path

#######################
# SAGITTAL: x + 20    #
# CORONAL: y + 0      #
# HORIZONTAL: z + 10  #
#######################

# --- Get the directory of the script ---
script_dir = Path(__file__).parent.resolve()

# --- Build the input and output directories ---
intput_directory = script_dir / ".." / "StlFiles" / "brain.stl"
intput_directory_str = str(intput_directory)
output_directory = script_dir / ".." / "MshFiles" / "test.msh"
output_directory_str = str(output_directory)

# --- Load the stl file and check if it is "closed" ---
mesh = trimesh.load(intput_directory_str)
print(mesh.is_watertight)

# --- Define the cut plane ---
z_slice = mesh.centroid + np.array([0, 0, 0]) # add the array for an offset
slice_3d = mesh.section(plane_normal=[0, 1, 0], 
                        plane_origin=z_slice)

# --- Transform into 2D ---
slice_2d, to_3d = slice_3d.to_planar()

# --- Print the number of polygons ---
print(f"Number of contours: {len(slice_2d.polygons_full)}")

# --- Plot the section ---
slice_2d.show()

# --- Get the contour of every polygon ---
all_polygons = slice_2d.polygons_full[:-1]
curve_loops = []
physical_boundaries = {"exterior": [], "holes": [], "white_matter": []}
scale_factor = 1.0

# --- Set characteristic length ---
lc = 1.0

# --- Initialize a gmsh object ---
gmsh.initialize()
gmsh.model.add("modelName")

# # --- Set options to have a more refined mesh on the boundaries and enable us to use a larger lc ---
# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
# gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 10) 
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8) # lower bound for the mesh size
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.2) # upper bound for the mesh size
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

# --- Loop over the polygons ---
surfaces = []
white_matter_surfaces = []
cl_poly_4 = None


for i, poly in enumerate(all_polygons):
    # --- Extract the coordinates ---
    poly_sim = poly.simplify(tolerance=0.1, preserve_topology=True) # Simplify only if needed
    coords = np.array(poly_sim.exterior.coords) * scale_factor
    
    # --- Create the gmsh points for this specific polygon ---
    poly_point_tags = []
    for c in coords[:-1]:
        p_tag = gmsh.model.geo.addPoint(c[0], c[1], 0, lc)
        poly_point_tags.append(p_tag)
    
    # --- Add the (closed) BSpline for this polygon ---
    l_tag_ext = gmsh.model.geo.addBSpline(poly_point_tags + [poly_point_tags[0]])
    cl_ext = gmsh.model.geo.addCurveLoop([l_tag_ext])

    if i == 4:
        cl_poly_4 = cl_ext
    
    # --- Create a curve loop for this polygon ---
    current_surface_loops = [cl_ext]
    physical_boundaries["exterior"].append(l_tag_ext)

    # --- White matter ---
    for j, interior in enumerate(poly_sim.interiors):
        coords_int = np.array(interior.coords) * scale_factor
        p_tags_int = []
        for c in coords_int[:-1]:
            p_tag = gmsh.model.geo.addPoint(c[0], c[1], 0, lc)
            p_tags_int.append(p_tag)
        
        l_tag_int = gmsh.model.geo.addBSpline(p_tags_int + [p_tags_int[0]])
        cl_int = gmsh.model.geo.addCurveLoop([l_tag_int])

        wm_loops = [cl_int]
        if cl_poly_4 is not None:
            wm_loops.append(wm_tag)

        # --- Fill the hole with white matter ---
        wm_tag = gmsh.model.geo.addPlaneSurface([cl_int])
        white_matter_surfaces.append(wm_tag)

        # --- Subtract the white matter to the full brain geometry ---
        current_surface_loops.append(cl_int)
        physical_boundaries["white_matter"].append(l_tag_int)


    # --- Build the total geometry ---
    s_tag = gmsh.model.geo.addPlaneSurface(current_surface_loops)
    surfaces.append(s_tag)
    
# --- Build the surface ---
gmsh.model.geo.synchronize()

# --- Physical group 1 (domain) ---
all_domain_surfaces = surfaces# + white_matter_surfaces
gmsh.model.addPhysicalGroup(2, surfaces, tag=1)
gmsh.model.setPhysicalName(2, 1, "gray_matter")

# Nuova Physical Group per la materia bianca
if physical_boundaries["white_matter"]:
    gmsh.model.addPhysicalGroup(2, white_matter_surfaces, tag=2)
    gmsh.model.setPhysicalName(2, 2, "white_matter")

# --- Physical group 2 (external boundary) ---
gmsh.model.addPhysicalGroup(1, physical_boundaries["exterior"], tag=10)
gmsh.model.setPhysicalName(1, 10, "exterior_boundary")

# --- Physical group 3 (internal holes) ---
if physical_boundaries["holes"]:
    gmsh.model.addPhysicalGroup(1, physical_boundaries["holes"], tag=11)
    gmsh.model.setPhysicalName(1, 11, "internal_boundaries")

# --- generate the 2D mesh ---
gmsh.model.mesh.generate(2)

# --- Plot the mesh ---
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

# --- Save it ---
gmsh.write(output_directory_str)
gmsh.finalize()

# --- Final print ---
print("Mesh succesfully generated!")
