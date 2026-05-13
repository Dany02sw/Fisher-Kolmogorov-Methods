import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from dolfin import plot

from Utilities.EnumUtilities import SpaceMethod, TimeMethod
from Utilities.DictionaryUtilities import ERROR_LABELS

# Function to plot the mesh _________________________________________________________________
def plot_mesh(mesh, title="Mesh", figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plot(mesh, title=title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# Function to print spatial convergence _____________________________________________________
POLY_COLORS = {
    1: "tab:red",
    2: "tab:purple",
    3: "tab:cyan",
    4: "yellowgreen",
    5: "tab:orange",
    6: "tab:brown",
}  # One color per polynomial degree, used consistently across all convergence plots

def plot_spatial_convergence(hs, err_c, err_grad, l, method=SpaceMethod.LDG, save=False):
    """
    Plots spatial convergence for a single polynomial degree l.
    """
    label_c, label_grad = ERROR_LABELS[method]

    hs   = np.array(hs)
    Ec   = np.array(err_c)
    Egrad = np.array(err_grad)
    col  = POLY_COLORS.get(l, "black")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_c, ax_grad = axes

    ax_c.loglog(hs, Ec,    "o-",  color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
    ax_c.loglog(hs, Ec[0] * (hs / hs[0]) ** (l + 1), "k--", linewidth=1.5, alpha=0.6, label=fr"$O(h^{{{l+1}}})$")

    ax_grad.loglog(hs, Egrad, "s--", color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
    ax_grad.loglog(hs, Egrad[0] * (hs / hs[0]) ** l, "k--", linewidth=1.5, alpha=0.6, label=fr"$O(h^{{{l}}})$")

    ax_c.set_title(fr"Errors ${label_c}$", fontsize=16)
    ax_c.set_xlabel(r"$h\;[-]$", fontsize=14)
    ax_c.set_ylabel(fr"$\|{label_c}\|_{{L^2(\Omega)}}$", fontsize=14)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12)
    ax_c.set_xlim(hs.min(),       hs.max())
    ax_c.set_ylim(Ec.min()*0.5,   Ec.max()*2.0)

    ax_grad.set_title(fr"Errors ${label_grad}$", fontsize=16)
    ax_grad.set_xlabel(r"$h\;[-]$", fontsize=14)
    ax_grad.set_ylabel(fr"$\|{label_grad}\|_{{L^2(\Omega)^d}}$", fontsize=14)
    ax_grad.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_grad.legend(fontsize=12)
    ax_grad.set_xlim(hs.min(),        hs.max())
    ax_grad.set_ylim(Egrad.min()*0.5, Egrad.max()*2.0)

    plt.tight_layout()

    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(plot_dir / f"space_convergence_l{l}_{timestamp}.png", dpi=300, bbox_inches="tight")
        print(f"Space convergence plot for l={l} saved into Plots folder!")

    plt.show()


# Function to plot polynomial convergence ___________________________________________________
def plot_polynomial_convergence(errors_c, errors_grad, l_list, h, method=SpaceMethod.LDG, save=False):
    """
    Function to plot polynomial convergence of both a base variable and a gradient-based one
    """
    label_c, label_grad = ERROR_LABELS[method]

    L_ext    = np.arange(l_list[0], l_list[-1] + 1)
    L_num    = np.array(l_list)
    Ec_vals  = np.array(errors_c)
    Eg_vals  = np.array(errors_grad)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(L_num, Ec_vals, "-o", color="yellowgreen", linewidth=2, markersize=8, label=fr"${label_c}$")
    ax.semilogy(L_num, Eg_vals, "--s", color="lime",       linewidth=2, markersize=8, label=fr"${label_grad}$")

    ref = Ec_vals[0] * (h ** (L_ext - l_list[0]))
    ax.semilogy(L_ext, ref, "k--", linewidth=3, label=r"$h^{\ell}$")

    ax.set_xlim(l_list[0], l_list[-1])
    ax.set_xticks(np.arange(l_list[0], l_list[-1]+1))
    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)
    ax.set_title(fr"Errors ${label_c}$ and ${label_grad}$", fontsize=20)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(fontsize=14, loc="upper right")
    plt.tight_layout()

    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(plot_dir / f"polynomial_convergence_{timestamp}.png", dpi=300, bbox_inches="tight")
        print("Polynomial convergence plot saved into Plots folder")

    plt.show()


# Function to print temporal convergence ____________________________________________________
BDF_COLORS = {
    1: "yellowgreen",
    2: "deepskyblue",
    3: "blue",
    4: "magenta",
    5: "crimson",
    6: "darkorange",
}

THETA_COLORS = {
    0.0: "tab:gold",         # explicit
    0.5: "deeppink",         # Crank-Nicolson
    1.0: "cornflowerblue",   # implicit
}

TIME_COLORS = {
    TimeMethod.BDF:   BDF_COLORS,
    TimeMethod.THETA: THETA_COLORS,
}

def plot_time_convergence(dt_list, err_c, err_grad, order, method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False):
    """
    Plots time convergence for BDF or Theta-method.
    """
    label_c, label_grad = ERROR_LABELS[space_method]

    dt_arr = np.array(dt_list, dtype=float)
    Ec     = np.array(err_c,    dtype=float)
    Egrad  = np.array(err_grad, dtype=float)

    col            = TIME_COLORS[method].get(order, "black")
    label          = fr"BDF{order}" if method == TimeMethod.BDF else fr"$\theta = {order}$"
    expected_order = order if method == TimeMethod.BDF else (2 if abs(order - 0.5) < 1e-10 else 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_c, ax_grad = axes

    ax_c.loglog(dt_arr, Ec,    "o-",  color=col, linewidth=2, markersize=7, label=label)
    ax_c.loglog(dt_arr, Ec[0] * (dt_arr / dt_arr[0]) ** expected_order, "k--", linewidth=1.5, alpha=0.6, label=r"$\tau^{" + str(expected_order) + "}$")

    ax_grad.loglog(dt_arr, Egrad, "s--", color=col, linewidth=2, markersize=7, label=label)
    ax_grad.loglog(dt_arr, Egrad[0] * (dt_arr / dt_arr[0]) ** expected_order, "k--", linewidth=1.5, alpha=0.6, label=r"$\tau^{" + str(expected_order) + "}$")

    ax_c.set_title(fr"Errors ${label_c}$ — {label}", fontsize=16)
    ax_c.set_xlabel(r"$\tau\;[-]$", fontsize=14)
    ax_c.set_ylabel(fr"$\|{label_c}\|_{{L^2(\Omega)}}$", fontsize=14)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12, loc="upper left")
    ax_c.set_xlim(dt_arr.min(),   dt_arr.max())
    ax_c.set_ylim(Ec.min()*0.5,   Ec.max()*2.0)

    ax_grad.set_title(fr"Errors ${label_grad}$ — {label}", fontsize=16)
    ax_grad.set_xlabel(r"$\tau\;[-]$", fontsize=14)
    ax_grad.set_ylabel(fr"$\|{label_grad}\|_{{L^2(\Omega)^d}}$", fontsize=14)
    ax_grad.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_grad.legend(fontsize=12, loc="upper left")
    ax_grad.set_xlim(dt_arr.min(),    dt_arr.max())
    ax_grad.set_ylim(Egrad.min()*0.5, Egrad.max()*2.0)

    plt.tight_layout()

    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = f"BDF{order}" if method == TimeMethod.BDF else f"THETA{order}"
        plt.savefig(plot_dir / f"time_convergence_{suffix}_{timestamp}.png", dpi=300, bbox_inches="tight")
        print("Time convergence plot saved into Plots folder!")

    plt.show()