from PlotMeshConfig.meshTest3 import MESH_GRID_CONFIGS, MESH_GRID_NAME
from MeshPlotUtilities        import plot_mesh_grid


# ── Mesh grid parameters ──────────────────────────────────────────────────────
PANEL_SIZE = (5, 5)   # (width_inches, height_inches) per panel
SAVE       = True    # set True to export the figure to Plots/


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_mesh_grid(
        mesh_configs     = MESH_GRID_CONFIGS,
        figsize_per_mesh = PANEL_SIZE,
        name             = MESH_GRID_NAME,
        save             = SAVE,
    )