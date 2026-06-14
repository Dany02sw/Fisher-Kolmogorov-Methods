from dolfin  import *
from mshr    import *
from pathlib import Path
from typing  import Union, Optional
from dolfin  import plot as dolfin_plot

import matplotlib.pyplot as plt

from fisher_kolmogorov.utilities.enum_utilities import BrainSection, MeshStructure, MeshType


# Print utility ___________________________________________________________________________________________________________________________________________________
def _print_mesh_info(
        mesh : Mesh
    ) -> None:
    """
    Simple helper method to print useful quantities about the mesh.
    """
    topology  = mesh.topology()
    geometry  = mesh.geometry()
    dim       = topology.dim()
    gdim      = geometry.dim()

    num_verts    = mesh.num_vertices()
    num_cells    = mesh.num_cells()
    num_edges    = topology.size(1) if dim >= 2 else 0
    num_faces    = topology.size(2) if dim >= 3 else 0

    coords       = mesh.coordinates()
    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
    Lx, Ly       = x_max - x_min, y_max - y_min
    area         = assemble(Constant(1.0) * dx(mesh))

    h_min, h_max = mesh.hmin(), mesh.hmax()
    h_avg        = (h_min + h_max) / 2.0
    h_ratio      = h_max / h_min if h_min > 0 else float('inf')

    cell_type    = mesh.ufl_cell()
    cell_volume  = area / num_cells

    print(f"\n  Mesh : {mesh.name()}")
    print(f"  {'─'*50}")
    print(f"  Topology")
    print(f"    Topological dim   : {dim}")
    print(f"    Geometrical dim   : {gdim}")
    print(f"    Cell type         : {cell_type}")
    print(f"    Num vertices      : {num_verts}")
    print(f"    Num edges         : {num_edges}")
    print(f"    Num cells         : {num_cells}")
    if dim == 3:
        print(f"    Num faces         : {num_faces}")
    print(f"  {'─'*50}")
    print(f"  Geometry")
    print(f"    Bounding box x    : [{x_min:.4f}, {x_max:.4f}]  (Lx = {Lx:.4f})")
    print(f"    Bounding box y    : [{y_min:.4f}, {y_max:.4f}]  (Ly = {Ly:.4f})")
    print(f"    Total area        : {area:.6f}")
    print(f"  {'─'*50}")
    print(f"  Cell size")
    print(f"    h_min             : {h_min:.6f}")
    print(f"    h_max             : {h_max:.6f}")
    print(f"    h_avg             : {h_avg:.6f}")
    print(f"    h_max / h_min     : {h_ratio:.4f}")
    print(f"    avg cell area     : {cell_volume:.6e}")
    print()


# Plot utility ___________________________________________________________________________________________________________________________________________________
def _plot_mesh(
        mesh    : Mesh,
        title   : str   = "Mesh",
        figsize : tuple = (10, 8) 
    ) -> None:
    """
    Helper function to plot the mesh at the beginning of the simulation if specified.
    """
    plt.figure(figsize=figsize)
    dolfin_plot(mesh, title=title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# Specific mesh constructors _____________________________________________________________________________________________________________________________________
def create_unit_square_mesh(
    N         : int,
    structure : MeshStructure = MeshStructure.STRUCTURED,
) -> Mesh:
    """
    Create a unit square mesh on [0, 1]^2.

    Parameters
    ----------
    N         : int
        Refinement parameter (number of subdivisions per side for structured
        meshes, or target number of cells for unstructured meshes).
    structure : MeshStructure
        STRUCTURED uses UnitSquareMesh; UNSTRUCTURED uses mshr.

    Returns
    -------
    mesh : Mesh
    """
    if structure == MeshStructure.UNSTRUCTURED:
        domain = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
        mesh   = generate_mesh(domain, N)
        mesh.rename("unstructured_square", "unstructured_square")
    else:
        mesh   = UnitSquareMesh(MPI.comm_self, N, N)
        mesh.rename("structured_square", "structured_square")

    return mesh


def create_rectangle_mesh(
    N         : int,
    P1        : Point,
    P2        : Point,
    structure : MeshStructure = MeshStructure.STRUCTURED,
) -> Mesh:
    """
    Create a rectangular mesh on [P1, P2].

    Parameters
    ----------
    N         : int
        Refinement parameter (subdivisions per unit length for structured
        meshes, or target number of cells for unstructured meshes).
    P1        : Point
        Lower-left corner of the rectangle.
    P2        : Point
        Upper-right corner of the rectangle.
    structure : MeshStructure
        STRUCTURED uses RectangleMesh; UNSTRUCTURED uses mshr.

    Returns
    -------
    mesh : Mesh
    """
    if structure == MeshStructure.UNSTRUCTURED:
        domain = Rectangle(P1, P2)
        mesh   = generate_mesh(domain, N)
        mesh.rename("unstructured_rectangle", "unstructured_rectangle")
    else:
        nx   = int(abs(P2[0] - P1[0])) * N
        ny   = int(abs(P2[1] - P1[1])) * N
        mesh = RectangleMesh(P1, P2, nx, ny)
        mesh.rename("structured_rectangle", "structured_rectangle")

    return mesh


def create_brain_mesh_2d(
    plane : BrainSection = BrainSection.SAGITTAL,
) -> tuple:
    """
    Load a 2-D brain mesh from an XDMF file bundled with the package.

    Parameters
    ----------
    plane : BrainSection
        Anatomical section to load: SAGITTAL, CORONAL, or HORIZONTAL.

    Returns
    -------
    mesh       : Mesh
    subdomains : MeshFunction
        Cell subdomains read from the XDMF file.
    """
    script_dir = Path(__file__).parent.resolve()
    mesh_dict = {
        BrainSection.SAGITTAL  : ("sagittal",   "test-sagittal.xdmf",   "test-sagittal.msh"),
        BrainSection.CORONAL   : ("coronal",    "test-coronal.xdmf",    "test-coronal.msh"),
        BrainSection.HORIZONTAL: ("horizontal", "test-horizontal.xdmf", "test-horizontal.msh"),
    }
    if plane not in mesh_dict:
        raise ValueError(f"Unknown brain plane: {plane}")

    mesh_name, xdmf_filename, msh_filename = mesh_dict[plane]

    mesh_src_dir = script_dir / "brain_meshes" / "mesh_src"
    msh_dir      = script_dir / "brain_meshes" / "msh_files"
    xdmf_path    = mesh_src_dir / xdmf_filename
    msh_path     = msh_dir / msh_filename

    # Fallback: if xdmf is missing, try to convert from msh
    if not xdmf_path.exists():
        if not msh_path.exists():
            raise FileNotFoundError(
                f"No mesh found for '{mesh_name}'.\n"
                f"  Expected XDMF : {xdmf_path}\n"
                f"  Expected MSH  : {msh_path}\n"
                f"The package installation may be incomplete; try reinstalling."
            )
        print(f"[INFO] XDMF not found, converting from MSH: {msh_path.name}...")
        from fisher_kolmogorov.meshes.brain_meshes.converter.gmsh_to_fenics import msh_to_xdmf  # local import to avoid circular deps
        msh_to_xdmf(msh_path, mesh_src_dir)

    return _load_xdmf_mesh(xdmf_path, mesh_name)


def create_user_mesh(
    msh_path : Union[Path, str],
) -> tuple:
    """
    Load a user-supplied mesh, converting from .msh to .xdmf if needed.

    The converted .xdmf and .h5 files are written to ``fk-xdmf/`` in the
    current working directory, which is created automatically if absent.
    Conversion is always performed (no caching), so stale files are never
    loaded silently.

    Parameters
    ----------
    msh_path : Path or str
        Path to the input .msh file.

    Returns
    -------
    mesh       : Mesh
    subdomains : MeshFunction or None
        Subdomains if present in the file, None otherwise.
    """
    msh_path  = Path(msh_path)
    if not msh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {msh_path}")

    output_dir = Path.cwd() / "fk-xdmf"
    output_dir.mkdir(exist_ok=True)

    from fisher_kolmogorov.meshes.brain_meshes.converter.gmsh_to_fenics import msh_to_xdmf  # local import to avoid circular deps
    xdmf_path = msh_to_xdmf(msh_path, output_dir)

    return _load_xdmf_mesh(xdmf_path, msh_path.stem)


# Helper XDMF loader (shared by BRAIN_2D and USER) ______________________________________________________________________________________________________________
def _load_xdmf_mesh(
        xdmf_path: Path,
        mesh_name: str
    ) -> tuple:
    """
    Read a mesh and its subdomains from an XDMF file.

    Parameters
    ----------
    xdmf_path : Path
        Path to the .xdmf file.
    mesh_name : str
        Name assigned to the loaded mesh object.

    Returns
    -------
    mesh       : Mesh
    subdomains : MeshFunction or None
    """
    mesh = Mesh()
    try:
        with XDMFFile(str(xdmf_path)) as infile:
            infile.read(mesh)
            subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
            try:
                infile.read(subdomains, "subdomains")
            except Exception:
                # Subdomains are optional for user-supplied meshes
                subdomains = None
        mesh.rename(mesh_name, mesh_name)
    except Exception as e:
        raise RuntimeError(f"Error loading mesh '{mesh_name}': {e}")

    return mesh, subdomains


# Core function: Mesh factory _____________________________________________________________________________________________________________________________________
def mesh_factory(
    mesh_type      : MeshType,
    N              : int              = None,
    structure      : MeshStructure    = MeshStructure.STRUCTURED,
    P1             : Point            = None,
    P2             : Point            = None,
    brain_plane    : BrainSection     = BrainSection.SAGITTAL,
    user_mesh_path : Union[Path, str] = None,
    show_plot      : bool             = False,
) -> tuple:
    """
    Unified factory that creates a mesh by dispatching to the appropriate
    constructor. Prefer the specific ``create_*`` functions for clarity in
    scripts and notebooks; use this entry point when the mesh type is
    determined at runtime.

    Parameters
    ----------
    mesh_type      : MeshType
        Which mesh to build.
    N              : int, optional
        Refinement parameter (ignored for BRAIN_2D and USER).
    structure      : MeshStructure
        STRUCTURED or UNSTRUCTURED (ignored for BRAIN_2D and USER).
    P1             : Point, optional
        Lower-left corner, required for RECTANGLE.
    P2             : Point, optional
        Upper-right corner, required for RECTANGLE.
    brain_plane    : BrainSection
        Anatomical section, used only for BRAIN_2D.
    user_mesh_path : Path or str, optional
        Path to a .msh file, required for USER.
    show_plot      : bool
        If True, display the mesh after creation.

    Returns
    -------
    mesh       : Mesh
    subdomains : MeshFunction or None
        Subdomains are returned for BRAIN_2D and USER (when present);
        None otherwise.
    """
    subdomains = None

    if mesh_type == MeshType.UNIT_SQUARE:
        mesh = create_unit_square_mesh(N=N, structure=structure)

    elif mesh_type == MeshType.RECTANGLE:
        if P1 is None or P2 is None:
            raise ValueError("P1 and P2 must be provided for MeshType.RECTANGLE")
        mesh = create_rectangle_mesh(N=N, P1=P1, P2=P2, structure=structure)

    elif mesh_type == MeshType.BRAIN_2D:
        mesh, subdomains = create_brain_mesh_2d(plane=brain_plane)

    elif mesh_type == MeshType.USER:
        if user_mesh_path is None:
            raise ValueError("user_mesh_path must be provided for MeshType.USER")
        mesh, subdomains = create_user_mesh(msh_path=user_mesh_path)

    else:
        raise ValueError(f"Unknown MeshType: {mesh_type}")

    _print_mesh_info(mesh)

    if show_plot:
        _plot_mesh(mesh=mesh, title=mesh.name())

    return mesh, subdomains
