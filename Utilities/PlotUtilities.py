import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from dolfin import plot

from Utilities.EnumUtilities import TimeMethod

# Function to plot the mesh _________________________________________________________________
def plot_mesh(mesh, title="Mesh", figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plot(mesh, title=title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# Function to plot polynomial convergence ___________________________________________________
def plot_polynomial_convergence(errors_c, errors_sigma, l_list, h, save=False):
    """
    Function to plot polynomial convergence of both a base variable and a gradient-based one
    """

    # ℓ dynamically adapted
    L_ext = np.arange(l_list[0], l_list[-1] + 1)

    # Extract numerical values
    L_num = np.array(l_list)
    Ec_vals = np.array(errors_c)
    Es_vals = np.array(errors_sigma)

    # Create the layout
    fig, ax = plt.subplots(figsize=(10, 6))

    # E_c
    ax.semilogy(
        L_num, Ec_vals, "-o", color="yellowgreen", linewidth=2,
        markersize=8, label=r"$E_c$-error"
    )

    # E_sigma 
    ax.semilogy(
        L_num, Es_vals, "--s", color="lime", linewidth=2,
        markersize=8, label=r"$E_{\sigma}$-error"
    )

    # h^ℓ
    ref = Ec_vals[0] * (h ** (L_ext - l_list[0]))
    ax.semilogy(
        L_ext, ref, "k--", linewidth=3, label=r"$h^{\ell}$"
    )

    # Aesthetics
    ax.set_xlim(l_list[0], l_list[-1])
    ax.set_xticks(np.arange(l_list[0], l_list[-1]+1))
    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)
    ax.set_title(r"Errors $E_c$ and $E_{\sigma}$", fontsize=20)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(fontsize=14, loc="upper right")
    plt.tight_layout()

    # Save figure
    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(plot_dir / f"polynomial_convergence_{timestamp}.png", dpi=300, bbox_inches="tight")
        print("Polynomial convergence plot saved into Plots folder")
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

def plot_spatial_convergence(hs, err_c, err_s, l, save=False):
    """
    Plots spatial convergence of E_c and E_sigma for a single polynomial degree l.
    """
    hs  = np.array(hs)
    Ec  = np.array(err_c)
    Es  = np.array(err_s)
    col = POLY_COLORS.get(l, "black")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_c, ax_s = axes

    # E_c — circle + solid
    ax_c.loglog(hs, Ec, "o-", color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
    ref_c = Ec[0] * (hs / hs[0]) ** (l + 1)
    ax_c.loglog(hs, ref_c, "k--", linewidth=1.5, alpha=0.6, label=fr"$O(h^{{{l+1}}})$")

    # E_sigma — square + dashed
    ax_s.loglog(hs, Es, "s--", color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
    ref_s = Es[0] * (hs / hs[0]) ** l
    ax_s.loglog(hs, ref_s, "k--", linewidth=1.5, alpha=0.6, label=fr"$O(h^{{{l}}})$")

    # Aesthetics E_c
    ax_c.set_title(r"Errors $E_c$", fontsize=16)
    ax_c.set_xlabel(r"$h\;[-]$", fontsize=14)
    ax_c.set_ylabel(r"$\|c(\cdot,T) - c_h\|_{L^2(\Omega)}$", fontsize=14)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12)
    ax_c.set_xlim(hs.min(),     hs.max())
    ax_c.set_ylim(Ec.min()*0.5, Ec.max()*2.0)

    # Aesthetics E_sigma
    ax_s.set_title(r"Errors $E_{\sigma}$", fontsize=16)
    ax_s.set_xlabel(r"$h\;[-]$", fontsize=14)
    ax_s.set_ylabel(r"$\|\sigma(\cdot,T) - \sigma_h\|_{L^2(\Omega)^d}$", fontsize=14)
    ax_s.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_s.legend(fontsize=12)
    ax_s.set_xlim(hs.min(),     hs.max())
    ax_s.set_ylim(Es.min()*0.5, Es.max()*2.0)

    plt.tight_layout()

    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(plot_dir / f"space_convergence_l{l}_{timestamp}.png", dpi=300, bbox_inches="tight")
        print(f"Space convergence plot for l={l} saved into Plots folder!")

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
    0.0: "tab:gold",       # explicit
    0.5: "tab:deeppink",     # Crank-Nicolson
    1.0: "cornflowerblue",   # implicit
}

TIME_COLORS = {
    TimeMethod.BDF:   BDF_COLORS,
    TimeMethod.THETA: THETA_COLORS,
}

def plot_time_convergence(dt_list, err_c, err_s, order, method=TimeMethod.BDF, save=False):
    """
    Plots time convergence of E_c and E_sigma for BDF or Theta-method.

    @param dt_list : list of time steps
    @param err_c   : list of E_c values
    @param err_s   : list of E_sigma values
    @param order   : BDF order (int) or theta value (float)
    @param method  : TimeMethod.BDF or TimeMethod.THETA
    @param save    : whether to save the figure
    """
    dt_arr  = np.array(dt_list, dtype=float)
    Ec      = np.array(err_c,   dtype=float)
    Es      = np.array(err_s,   dtype=float)

    col   = TIME_COLORS[method].get(order, "black")
    label = fr"BDF{order}" if method == TimeMethod.BDF else fr"$\theta = {order}$"
    expected_order = order if method == TimeMethod.BDF else (2 if abs(order - 0.5) < 1e-10 else 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_c, ax_s = axes

    # E_c — circle + solid
    ax_c.loglog(dt_arr, Ec, "o-", color=col, linewidth=2, markersize=7, label=label)
    ref_c = Ec[0] * (dt_arr / dt_arr[0]) ** expected_order
    ax_c.loglog(dt_arr, ref_c, "k--", linewidth=1.5, alpha=0.6,
                label=r"$\tau^{" + str(expected_order) + "}$")

    # E_sigma — square + dashed
    ax_s.loglog(dt_arr, Es, "s--", color=col, linewidth=2, markersize=7, label=label)
    ref_s = Es[0] * (dt_arr / dt_arr[0]) ** expected_order
    ax_s.loglog(dt_arr, ref_s, "k--", linewidth=1.5, alpha=0.6,
                label=r"$\tau^{" + str(expected_order) + "}$")

    # Aesthetics E_c
    ax_c.set_title(fr"Errors $E_c$ — {label}", fontsize=16)
    ax_c.set_xlabel(r"$\tau\;[-]$", fontsize=14)
    ax_c.set_ylabel(r"$\|c(\cdot,T) - c_h\|_{L^2(\Omega)}$", fontsize=14)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12, loc="upper left")
    ax_c.set_xlim(dt_arr.min(), dt_arr.max())
    ax_c.set_ylim(Ec.min()*0.5, Ec.max()*2.0)

    # Aesthetics E_sigma
    ax_s.set_title(fr"Errors $E_{{\sigma}}$ — {label}", fontsize=16)
    ax_s.set_xlabel(r"$\tau\;[-]$", fontsize=14)
    ax_s.set_ylabel(r"$\|\sigma(\cdot,T) - \sigma_h\|_{L^2(\Omega)^d}$", fontsize=14)
    ax_s.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_s.legend(fontsize=12, loc="upper left")
    ax_s.set_xlim(dt_arr.min(), dt_arr.max())
    ax_s.set_ylim(Es.min()*0.5, Es.max()*2.0)

    plt.tight_layout()

    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = f"BDF{order}" if method == TimeMethod.BDF else f"THETA{order}"
        plt.savefig(plot_dir / f"time_convergence_{suffix}_{timestamp}.png", dpi=300, bbox_inches="tight")
        print("Time convergence plot saved into Plots folder!")

    plt.show()