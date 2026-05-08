from dolfin import *
from mshr import *
import matplotlib.pylab as plt
from pathlib import Path
from EnumUtilities import BrainSection

# Square mesh generator _________________________________________________________________________
def create_unite_square_mesh(N, unstructured=False, plots=False):
    if unstructured:
        domain = Rectangle(Point((0.0, 0.0)), Point((1.0, 1.0)))
        mesh = generate_mesh(domain, N)
        print("Unstructured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()
    else:
        mesh = UnitSquareMesh(MPI.comm_self, N, N)
        print("Structured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()

    return mesh

# Rectangle mesh generator ______________________________________________________________________
def create_rectangle_mesh(N, P1, P2, unstructured = False, plots = False):
    if unstructured:
        domain = Rectangle(P1, P2)
        mesh = generate_mesh(domain, N)
        print("Unstructured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()
    else:
        mesh = RectangleMesh(P1, P2, int(abs(P2[0] - P1[0]))*N, int(abs(P2[1] - P1[1]))*N)
        print("Structured mesh: ")
        print("-> hmax = ", mesh.hmax())
        print("-> hmin = ", mesh.hmin())
        print("-> Number of elements = ", mesh.num_cells())
        if plots:
            plot(mesh)
            plt.show()
    return mesh

# Import Brain mesh 2D ________________________________________________________________________
def importBrainMesh2D(plane=BrainSection.SAGITTAL, plotMesh=False):

    # Get the directory of the script 
    script_dir = Path(__file__).parent.resolve()

    # Build the correct directory 
    if plane == BrainSection.SAGITTAL:
        mesh_path = script_dir / "BrainMeshes" / "MeshSrc" / "test-sagittal.xdmf"
        mesh_name = "sagittal"

    elif plane == BrainSection.CORONAL:
        mesh_path = script_dir / "BrainMeshes" / "MeshSrc" / "test-coronal.xdmf"
        mesh_name = "coronal"

    elif plane == BrainSection.HORIZONTAL:
        print("loading horizontal section mesh...")
        mesh_path = script_dir / "BrainMeshes" / "MeshSrc" / "test-horizontal.xdmf"
        mesh_name = "horizontal"

    # Build an empty Mesh object 
    mesh = Mesh()

    # Read the .xdmf file 
    try:
        with XDMFFile(str(mesh_path)) as infile:
            infile.read(mesh)
            subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
            infile.read(subdomains, "subdomains")
        mesh.rename(mesh_name, mesh_name)
        print(f"2D brain {mesh_name} section mesh succesfully loaded!")
        print(f"Number of nodes: {mesh.num_vertices()}")
        print(f"Number of elements (triangles): {mesh.num_cells()}")
        print(f"Max cell size: {mesh.hmax()}")
        print(f"Min cell size: {mesh.hmin()}")
    except Exception as e:
        print(f"Error loading the mesh: {e}")

    # Plot the mesh 
    if plotMesh:
        plt.figure(figsize=(10, 8))
        plot(mesh, title="2D Brain section")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    return mesh, subdomains
