import matplotlib.pyplot as plt

from datetime import datetime
from dolfin   import plot as dolfin_plot

from config              import MESHES_DIR
from Meshes              import mesh_factory
from Plots.PlotUtilities import _save_plot

# =========================================================================================================================================
# Mesh grid plot ==========================================================================================================================
# =========================================================================================================================================
def plot_mesh_grid(mesh_configs: list, figsize_per_mesh=(5, 5), name="mesh_grid", save=False):
    """
    Plots multiple meshes in a single row of subplots, annotating each with
    h_min, h_max, and h_avg.

    Each entry in ``mesh_configs`` is a dict passed to ``mesh_factory``, e.g.::

        [
            {"mesh_type": MeshType.UNIT_SQUARE, "N": 8},
            {"mesh_type": MeshType.RECTANGLE,   "N": 4,
             "P1": Point(0,0), "P2": Point(2,1)},
            {"mesh_type": MeshType.BRAIN_2D,    "brain_plane": BrainSection.SAGITTAL},
        ]

    Parameters
    ----------
    mesh_configs     : list of dict
        Each dict is forwarded as kwargs to ``mesh_factory``.  The keys
        ``show_plot`` is forced to False internally.
    figsize_per_mesh : tuple (width, height)
        Size in inches of each individual subplot panel.
    save             : bool
        If True, saves the figure to Plots/.
    """

    n = len(mesh_configs)
    if n == 0:
        raise ValueError("mesh_configs must contain at least one entry.")

    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_mesh[0] * n, figsize_per_mesh[1]),
        squeeze=False,
    )
    axes = axes[0]  # shape (n,)

    for ax, cfg in zip(axes, mesh_configs):
        kw          = {**cfg, "show_plot": False}
        mesh, _     = mesh_factory(**kw)

        h_min = mesh.hmin()
        h_max = mesh.hmax()
        h_avg = (h_min + h_max) / 2.0
        N_ele = mesh.num_cells()

        plt.sca(ax)
        dolfin_plot(mesh, title="")

        ax.set_title(mesh.name(), fontsize=12, fontweight="bold", pad=4)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        info_text = (
            fr"$h_{{\min}}={h_min:.6f}$"   "\n"
            fr"$h_{{\max}}={h_max:.6f}$"   "\n"
            fr"$h_{{\mathrm{{avg}}}}={h_avg:.6f}$"   "\n"
            fr"$N^\circ_{{\mathrm{{ele}}}}={N_ele}$"
        )
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="gray"),
        )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            MESHES_DIR / f"{name}_{timestamp}.png",
            "Mesh grid plot",
        )

    plt.show()