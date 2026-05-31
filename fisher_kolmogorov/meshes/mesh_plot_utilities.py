import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from datetime          import datetime
from matplotlib.collections import PolyCollection

from config              import MESHES_DIR
from fisher_kolmogorov.meshes.mesh_import   import mesh_factory
from fisher_kolmogorov.plots.plot_utilities import _save_plot

# Dictionary for brain subdomains _________________________________________________________________________________________________________
SUBDOMAIN_LABELS = {
    1: "Gray Matter",
    2: "White Matter",
}

# Private helper function to color subdomains _____________________________________________________________________________________________
def _plot_mesh_with_subdomains(ax: plt.Axes, mesh, subdomains) -> None:
    """
    Renders *mesh* on *ax* coloring each triangular cell by its subdomain tag.

    Cells are drawn as filled polygons via ``PolyCollection``; mesh edges are
    overlaid in a thin dark line.  A legend with one colored patch per unique
    tag is placed below the axes.

    Parameters
    ----------
    ax         : matplotlib Axes
        Target subplot.
    mesh       : dolfin.Mesh
    subdomains : dolfin.MeshFunction
        Cell-wise integer marker function.
    """
    coords  = mesh.coordinates()        # shape (n_vertices, 2)
    cells   = mesh.cells()              # shape (n_cells, 3)  — vertex indices
    tags    = subdomains.array()        # shape (n_cells,)

    unique_tags = np.unique(tags)
    n_tags      = len(unique_tags)
    cmap = plt.get_cmap("tab10" if n_tags <= 10 else "tab20")
    tag_to_color = {
        int(tag): cmap(i)
        for i, tag in enumerate(unique_tags)
    }

    # Build one PolyCollection per tag so each gets a uniform color
    for tag, color in tag_to_color.items():
        mask      = tags == tag
        triangles = coords[cells[mask]]   # shape (k, 3, 2)
        col       = PolyCollection(
            triangles,
            facecolors=color,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.85,
        )
        ax.add_collection(col)

    # Fit axes to mesh bounding box
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_aspect("equal")

    # Legend below the axes
    patches = []
    for tag in unique_tags:
        color = tag_to_color[int(tag)]
        label = SUBDOMAIN_LABELS.get(int(tag), "")
        label = f"{tag} — {label}" if label else str(tag)
        patches.append(mpatches.Patch(color=color, label=label))

    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(n_tags, 4),
        fontsize=8,
        framealpha=0.85,
        title="Subdomains",
        title_fontsize=8,
    )

# Function to plot general meshes _________________________________________________________________________________________________________
def plot_mesh_grid(mesh_configs: list, figsize_per_mesh=(5, 5), name="mesh_grid", save=False):
    """
    Plots multiple meshes in a single row of subplots, annotating each with
    h_min, h_max, h_avg and N_ele.

    If a mesh has subdomains (returned as the second value from ``mesh_factory``),
    cells are colored by subdomain tag and a legend is shown below the subplot.
    Otherwise the mesh is rendered with the default dolfin wireframe style.

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
        Each dict is forwarded as kwargs to ``mesh_factory``.
        The key ``show_plot`` is forced to False internally.
    figsize_per_mesh : tuple (width, height)
        Size in inches of each individual subplot panel.
    name             : str
        Base filename used when saving.
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
    axes = axes[0]

    for ax, cfg in zip(axes, mesh_configs):
        kw            = {**cfg, "show_plot": False}
        mesh, subdoms = mesh_factory(**kw)

        h_min = mesh.hmin()
        h_max = mesh.hmax()
        h_avg = (h_min + h_max) / 2.0
        N_ele = mesh.num_cells()

        if subdoms is not None:
            _plot_mesh_with_subdomains(ax, mesh, subdoms)
        else:
            plt.sca(ax)
            from dolfin import plot as dolfin_plot
            dolfin_plot(mesh, title="")

        ax.set_title(mesh.name(), fontsize=12, fontweight="bold", pad=4)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        info_text = (
            fr"$h_{{\min}}={h_min:.6f}$"                "\n"
            fr"$h_{{\max}}={h_max:.6f}$"                "\n"
            fr"$h_{{\mathrm{{avg}}}}={h_avg:.6f}$"      "\n"
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