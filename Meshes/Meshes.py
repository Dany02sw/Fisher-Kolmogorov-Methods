from dolfin import *
from mshr import *
import matplotlib.pylab as plt
from pathlib import Path

from Utilities.EnumUtilities import BrainSection
from Utilities.PlotUtilities import plot_mesh

# Utilitary private method to print mesh infos __________________________________________________
def _print_mesh_info(mesh):
    print(f"\n{mesh.name()}:")
    print(f"  Number of nodes:    {mesh.num_vertices()}")
    print(f"  Number of elements: {mesh.num_cells()}")
    print(f"  Min cell size:      {mesh.hmin():.6f}")
    print(f"  Max cell size:      {mesh.hmax():.6f}")

# Square mesh generator _________________________________________________________________________
def create_unite_square_mesh(N, unstructured=False, plotMesh=False):
    if unstructured:
        domain    = Rectangle(Point((0.0, 0.0)), Point((1.0, 1.0)))
        mesh      = generate_mesh(domain, N)
        mesh.rename("unstructured_square", "unstructured_square")

    else:
        mesh      = UnitSquareMesh(MPI.comm_self, N, N)
        mesh.rename("structured_square", "structured_square")

    _print_mesh_info(mesh)
    if plotMesh:
        plot_mesh(mesh=mesh, title=mesh.name())

    return mesh

# Rectangle mesh generator ______________________________________________________________________
def create_rectangle_mesh(N, P1, P2, unstructured=False, plotMesh=False):
    if unstructured:
        domain    = Rectangle(P1, P2)
        mesh      = generate_mesh(domain, N)
        mesh.rename("unstructured_rectangle", "unstructured_rectangle")

    else:
        mesh      = RectangleMesh(P1, P2, int(abs(P2[0] - P1[0]))*N, int(abs(P2[1] - P1[1]))*N)
        mesh.rename("structured_rectangle", "structured_rectangle")

    _print_mesh_info(mesh)
    if plotMesh:
        plot_mesh(mesh=mesh, title=mesh.name())

    return mesh

# Import Brain mesh 2D ________________________________________________________________________
def importBrainMesh2D(plane=BrainSection.SAGITTAL, plotMesh=False):

    # Get the directory of the script 
    script_dir = Path(__file__).parent.resolve()

    # Dictionary for name and file
    mesh_dict = {
        BrainSection.SAGITTAL:   ("sagittal",   "test-sagittal.xdmf"),
        BrainSection.CORONAL:    ("coronal",     "test-coronal.xdmf"),
        BrainSection.HORIZONTAL: ("horizontal",  "test-horizontal.xdmf"),
    }
    if plane not in mesh_dict:
        raise ValueError(f"Unknown plane: {plane}")
    
    # Build the correct directory 
    mesh_name, filename = mesh_dict[plane]
    mesh_path = script_dir / "BrainMeshes" / "MeshSrc" / filename

    # Build an empty Mesh object 
    mesh = Mesh()

    # Read the .xdmf file 
    try:
        with XDMFFile(str(mesh_path)) as infile:
            infile.read(mesh)
            subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
            infile.read(subdomains, "subdomains")

        mesh.rename(mesh_name, mesh_name)
        _print_mesh_info(mesh)

    except Exception as e:
        raise RuntimeError(f"Error loading mesh '{mesh_name}': {e}")

    # Plot the mesh 
    if plotMesh:
        plot_mesh(mesh=mesh, title=f"{mesh_name} section mesh")

    return mesh, subdomains
