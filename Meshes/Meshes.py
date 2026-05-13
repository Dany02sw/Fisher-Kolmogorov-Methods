from dolfin import *
from mshr import *
from pathlib import Path

from Utilities.EnumUtilities import BrainSection, MeshStructure, MeshType
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



# # ── Print utility ──────────────────────────────────────────────────────────────

# def _print_mesh_info(mesh):
#     topology  = mesh.topology()
#     geometry  = mesh.geometry()
#     dim       = topology.dim()
#     gdim      = geometry.dim()

#     num_verts    = mesh.num_vertices()
#     num_cells    = mesh.num_cells()
#     num_edges    = topology.size(1) if dim >= 2 else 0
#     num_faces    = topology.size(2) if dim >= 3 else 0

#     coords       = mesh.coordinates()
#     x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
#     x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
#     Lx, Ly       = x_max - x_min, y_max - y_min
#     area         = assemble(Constant(1.0) * dx(mesh))

#     h_min, h_max = mesh.hmin(), mesh.hmax()
#     h_avg        = area / num_cells
#     h_ratio      = h_max / h_min if h_min > 0 else float('inf')

#     cell_type    = mesh.ufl_cell()
#     cell_volume  = area / num_cells

#     print(f"\n  Mesh : {mesh.name()}")
#     print(f"  {'─'*50}")
#     print(f"  Topology")
#     print(f"    Topological dim   : {dim}")
#     print(f"    Geometrical dim   : {gdim}")
#     print(f"    Cell type         : {cell_type}")
#     print(f"    Num vertices      : {num_verts}")
#     print(f"    Num edges         : {num_edges}")
#     print(f"    Num cells         : {num_cells}")
#     if dim == 3:
#         print(f"    Num faces         : {num_faces}")
#     print(f"  {'─'*50}")
#     print(f"  Geometry")
#     print(f"    Bounding box x    : [{x_min:.4f}, {x_max:.4f}]  (Lx = {Lx:.4f})")
#     print(f"    Bounding box y    : [{y_min:.4f}, {y_max:.4f}]  (Ly = {Ly:.4f})")
#     print(f"    Total area        : {area:.6f}")
#     print(f"  {'─'*50}")
#     print(f"  Cell size")
#     print(f"    h_min             : {h_min:.6f}")
#     print(f"    h_max             : {h_max:.6f}")
#     print(f"    h_max / h_min     : {h_ratio:.4f}")
#     print(f"    avg cell area     : {cell_volume:.6e}")
#     print()

# # ── Mesh factory ───────────────────────────────────────────────────────────────

# def create_mesh(
#     mesh_type   : MeshType,
#     N           : int                = 10,
#     structure   : MeshStructure      = MeshStructure.STRUCTURED,
#     P1          : Point              = None,
#     P2          : Point              = None,
#     brain_plane : BrainSection       = BrainSection.SAGITTAL,
#     plot_mesh   : bool               = False,
# ):
#     """
#     Factory function that creates and returns the requested mesh.

#     Parameters
#     ----------
#     mesh_type   : MeshType         — which mesh to build
#     N           : int              — refinement parameter
#     structure   : MeshStructure    — structured vs unstructured (ignored for brain)
#     P1, P2      : Point            — corners for RECTANGLE (ignored otherwise)
#     brain_plane : BrainSection     — section for BRAIN_2D (ignored otherwise)
#     plot_mesh   : bool             — whether to plot after creation

#     Returns
#     -------
#     mesh   : Mesh
#     subdomains : MeshFunction | None   (only for BRAIN_2D, else None)
#     """

#     subdomains = None

#     # ── Unit square ────────────────────────────────────────────────────────────
#     if mesh_type == MeshType.UNIT_SQUARE:
#         if structure == MeshStructure.UNSTRUCTURED:
#             domain = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
#             mesh   = generate_mesh(domain, N)
#             mesh.rename("unstructured_square", "unstructured_square")
#         else:
#             mesh   = UnitSquareMesh(MPI.comm_self, N, N)
#             mesh.rename("structured_square", "structured_square")

#     # ── Rectangle ──────────────────────────────────────────────────────────────
#     elif mesh_type == MeshType.RECTANGLE:
#         if P1 is None or P2 is None:
#             raise ValueError("P1 and P2 must be provided for MeshType.RECTANGLE")
#         if structure == MeshStructure.UNSTRUCTURED:
#             domain = Rectangle(P1, P2)
#             mesh   = generate_mesh(domain, N)
#             mesh.rename("unstructured_rectangle", "unstructured_rectangle")
#         else:
#             nx   = int(abs(P2[0] - P1[0])) * N
#             ny   = int(abs(P2[1] - P1[1])) * N
#             mesh = RectangleMesh(P1, P2, nx, ny)
#             mesh.rename("structured_rectangle", "structured_rectangle")

#     # ── Brain 2D ───────────────────────────────────────────────────────────────
#     elif mesh_type == MeshType.BRAIN_2D:
#         script_dir = Path(__file__).parent.resolve()
#         mesh_dict  = {
#             BrainSection.SAGITTAL  : ("sagittal",    "test-sagittal.xdmf"),
#             BrainSection.CORONAL   : ("coronal",     "test-coronal.xdmf"),
#             BrainSection.HORIZONTAL: ("horizontal",  "test-horizontal.xdmf"),
#         }
#         if brain_plane not in mesh_dict:
#             raise ValueError(f"Unknown brain plane: {brain_plane}")

#         mesh_name, filename = mesh_dict[brain_plane]
#         mesh_path = script_dir / "BrainMeshes" / "MeshSrc" / filename
#         mesh      = Mesh()

#         try:
#             with XDMFFile(str(mesh_path)) as infile:
#                 infile.read(mesh)
#                 subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
#                 infile.read(subdomains, "subdomains")
#             mesh.rename(mesh_name, mesh_name)
#         except Exception as e:
#             raise RuntimeError(f"Error loading mesh '{mesh_name}': {e}")

#     else:
#         raise ValueError(f"Unknown MeshType: {mesh_type}")

#     _print_mesh_info(mesh)

#     if plot_mesh:
#         plot_mesh(mesh=mesh, title=mesh.name())

#     return mesh, subdomains
