import meshio
import sys

from pathlib import Path


# ===== Converter =====
def msh_to_xdmf(input_file, output_directory):
    """
    Converter: from .msh to .xdmf file
    """
    # --- Define the output file ---
    output_file = output_directory / f"{input_file.stem}.xdmf" # .stem erases the extension (.msh)

    # --- Create the output directory if it does not exists ---
    output_directory.mkdir(parents=True, exist_ok=True)

    # --- Load the mesh from gmsh ---
    print(f"Reading: {input_file.name}...")
    try:
        msh = meshio.read(str(input_file))

        # --- Extract the physical boundary ---
        triangle_data = {}
        if "gmsh:physical" in msh.cell_data_dict:
            if "triangle" in msh.cell_data_dict["gmsh:physical"]:
                triangle_data = {"subdomains": [msh.cell_data_dict["gmsh:physical"]["triangle"]]}

        # --- Check if there are some triangles ---
        cells = msh.get_cells_type("triangle")
        if len(cells) == 0:
            print(f"[ERROR]: No triangles found in the mesh file {input_file}!")
            sys.exit(1)

        # --- Extract the triangles and delete the lines ---
        triangle_mesh = meshio.Mesh(
            points=msh.points[:, :2], # Take only x and y for a 2d mesh 
            cells=[("triangle", cells)],
            cell_data=triangle_data
        )

        # --- Save the clean mesh for fenics ---
        meshio.write(str(output_file), triangle_mesh)
        print(f"Mesh cleaned and saved as {output_file.name} in {output_directory.name}!")
        return output_file
    
    except Exception as e:
        print(f"\n[ERROR] Could not process the mesh: {e}")
        sys.exit(1)

# ===== Interactive logic =====
def main():
    """
    Interactive logic converter
    """
    # --- Get the directory of the script ---
    script_dir = Path(__file__).parent.resolve()

    # --- Build the input and output directories ---
    input_directory  = script_dir / ".." / "msh_files"
    output_directory = script_dir / ".." / "mesh_src"

    # --- Check existence of input directory ---
    if not input_directory.exists():
        print(f"\n[FATAL ERROR]: Input directory does not exists!")
        print(f"Missing path: {input_directory.resolve()}")
        print("You should either place the folder 'MshFiles' in the correct position or run a MeshGenerator first.")
        sys.exit(1)

    # --- Print all files in the input directory ---
    msh_files = sorted(list(input_directory.glob("*.msh")))
    if not msh_files:
        print(f"\n[WARNING]: No .msh found inside: {input_directory.resolve()}")
        print("Add .msh and try again.")
        sys.exit(0)

    # --- Ask the user which one they wants to convert ---
    print("\n")
    print(36*"#")
    print("### Mesh converter: GMSH -> XDMF ###")
    print(36*"#")
    print("\nAvailable .msh files:")
    for i, file in enumerate(msh_files):
        print(f"[{i}] {file.name}")
    try:
        choice = int(input("\nInsert the number corresponding to the .msh file to clean: "))
        if choice < 0 or choice >= len(msh_files):
            raise ValueError
    except ValueError:
        print("[ERROR]: Invalid choice.")
        sys.exit(1)
    input_file = msh_files[choice]
    msh_to_xdmf(input_file, output_directory)

# ===== Main for running this script alone =====
if __name__ == "__main__":
    main()