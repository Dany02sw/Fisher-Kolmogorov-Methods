import meshio
from pathlib import Path

###############################################
# AUTOMATIZZARE IL PASSAGGIO DEL FILE DA PULIRE
###############################################

# --- Get the directory of the script ---
script_dir = Path(__file__).parent.resolve()

# --- Build the input and output directories ---
intput_directory = script_dir / ".." / "MshFiles" / "brain3d.msh"
intput_directory_str = str(intput_directory)
output_directory = script_dir / ".." / "MeshSrc" / "brain3d.xdmf"
output_directory_str = str(output_directory)

# --- Load the mesh from gmsh ---
msh = meshio.read(intput_directory_str)

# --- Extract the physical boundary ---
cell_data = {}
if "gmsh:physical" in msh.cell_data_dict:
    cell_data = {"subdomains": [msh.cell_data_dict["gmsh:physical"]["tetra"]]}

# --- Extract the tetrahedras and delete the lines ---
tetra_mesh = meshio.Mesh(
    points=msh.points, 
    cells=[("tetra", msh.get_cells_type("tetra"))],
    cell_data=cell_data

)

# --- Save the clean mesh for fenics ---
meshio.write(output_directory_str, tetra_mesh)

print("Mesh cleaned and saved into 'Test 3/MeshSrc/brain3d.xdmf'!")