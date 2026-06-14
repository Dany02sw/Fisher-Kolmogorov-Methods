import meshio
import sys

from pathlib import Path


def msh_to_xdmf(input_file: Path, output_directory: Path) -> Path:
    """
    Convert a Gmsh .msh file to a FEniCS-compatible .xdmf file.

    Only triangular cells are extracted. Physical group tags are preserved
    as subdomain markers under the key ``"subdomains"``.

    Parameters
    ----------
    input_file       : Path
        Path to the input .msh file.
    output_directory : Path
        Directory where the .xdmf (and companion .h5) file will be written.
        Created automatically if it does not exist.

    Returns
    -------
    output_file : Path
        Path to the generated .xdmf file.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / f"{input_file.stem}.xdmf"

    print(f"Reading: {input_file.name}...")
    try:
        msh = meshio.read(str(input_file))

        # Preserve physical group tags as subdomain markers
        triangle_data = {}
        if "gmsh:physical" in msh.cell_data_dict:
            if "triangle" in msh.cell_data_dict["gmsh:physical"]:
                triangle_data = {"subdomains": [msh.cell_data_dict["gmsh:physical"]["triangle"]]}

        cells = msh.get_cells_type("triangle")
        if len(cells) == 0:
            print(f"[ERROR]: No triangles found in the mesh file {input_file}!")
            sys.exit(1)

        # Drop z-coordinate for 2-D meshes
        triangle_mesh = meshio.Mesh(
            points=msh.points[:, :2],
            cells=[("triangle", cells)],
            cell_data=triangle_data,
        )

        meshio.write(str(output_file), triangle_mesh)
        print(f"Mesh saved as {output_file.name} in {output_directory.name}!")
        return output_file

    except Exception as e:
        print(f"\n[ERROR] Could not process the mesh: {e}")
        sys.exit(1)
