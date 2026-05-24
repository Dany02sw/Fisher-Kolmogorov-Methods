from dolfin import Point

from MeshPlotUtilities import plot_mesh_grid
from Utilities.EnumUtilities  import MeshType, MeshStructure, BrainSection


# ── Mesh grid parameters ──────────────────────────────────────────────────────

MESH_GRID_CONFIGS = [
    {
        "mesh_type" : MeshType.UNIT_SQUARE,
        "N"         : 8,
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.UNIT_SQUARE,
        "N"         : 16,
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 6,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(2.0, 1.0),
        "structure" : MeshStructure.STRUCTURED,
    },
    {
        "mesh_type"   : MeshType.BRAIN_2D,
        "brain_plane" : BrainSection.SAGITTAL,
    },
]

PANEL_SIZE = (5, 5)   # (width_inches, height_inches) per panel
SAVE       = False    # set True to export the figure to Plots/


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_mesh_grid(
        mesh_configs     = MESH_GRID_CONFIGS,
        figsize_per_mesh = PANEL_SIZE,
        save             = SAVE,
    )