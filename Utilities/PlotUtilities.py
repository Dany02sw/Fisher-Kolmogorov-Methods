import matplotlib.pyplot as plt
import numpy             as np

from pathlib  import Path
from datetime import datetime

from Utilities.EnumUtilities       import SpaceMethod, TimeMethod
from Utilities.DictionaryUtilities import ERROR_LABELS_PLOT, NORM_LABELS


# Function to print spatial convergence ___________________________________________________________________________________________________
POLY_COLORS = {
    1: "red",
    2: "darkmagenta",
    3: "darkcyan",
    4: "yellowgreen",
    5: "blue",
    6: "crimson",
    7: "gold",
    8: "lime"
}  # One color per polynomial degree, used consistently across all convergence plots

def plot_spatial_convergence(hs, err_c, err_grad, l, method=SpaceMethod.LDG, save=False):
    """
    Plots spatial convergence for a single polynomial degree l.
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[method]
    norm_label_c, norm_label_grad = NORM_LABELS[method]
 
    hs    = np.array(hs)
    Ec    = np.array(err_c)
    Egrad = np.array(err_grad)
    col   = POLY_COLORS.get(l, "black")
 
    fig, axes     = plt.subplots(1, 2, figsize=(15, 6))
    ax_c, ax_grad = axes
 
    ax_c.loglog(hs, Ec,    "o-",  color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
    ax_c.loglog(hs, Ec[0] * (hs / hs[0]) ** (l + 1), "k--", linewidth=1.5, alpha=0.6, label=fr"$O(h^{{{l+1}}})$")
 
    ax_grad.loglog(hs, Egrad, "s--", color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
    ax_grad.loglog(hs, Egrad[0] * (hs / hs[0]) ** l, "k--", linewidth=1.5, alpha=0.6, label=fr"$O(h^{{{l}}})$")
 
    ax_c.set_title(fr"Errors ${label_c}$", fontsize=16)
    ax_c.set_xlabel(r"$h\;[-]$", fontsize=14)
    ax_c.set_ylabel(norm_label_c, fontsize=14)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12)
    ax_c.set_xlim(hs.min(),       hs.max())
    ax_c.set_ylim(Ec.min()*0.5,   Ec.max()*2.0)
 
    ax_grad.set_title(fr"Errors ${label_grad}$", fontsize=16)
    ax_grad.set_xlabel(r"$h\;[-]$", fontsize=14)
    ax_grad.set_ylabel(norm_label_grad, fontsize=14)
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


# Function to plot polynomial convergence _________________________________________________________________________________________________
def plot_polynomial_convergence(errors_c, errors_grad, l_list, h, method=SpaceMethod.LDG, save=False):
    """
    Function to plot polynomial convergence of both a base variable and a gradient-based one
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[method]
    norm_label_c, norm_label_grad = NORM_LABELS[method]
 
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
    ax.set_ylabel(f"{norm_label_c}  /  {norm_label_grad}", fontsize=14)
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


# Function to print temporal convergence __________________________________________________________________________________________________
BDF_COLORS = {
    1: "yellowgreen",
    2: "deepskyblue",
    3: "blue",
    4: "deeppink",
    5: "crimson",
    6: "darkorange",
}

THETA_COLORS = {
    0.0: "gold",         # explicit
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
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]
 
    dt_arr = np.array(dt_list,  dtype=float)
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
    ax_c.set_ylabel(norm_label_c, fontsize=14)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12, loc="upper left")
    ax_c.set_xlim(dt_arr.min(),   dt_arr.max())
    ax_c.set_ylim(Ec.min()*0.5,   Ec.max()*2.0)
 
    ax_grad.set_title(fr"Errors ${label_grad}$ — {label}", fontsize=16)
    ax_grad.set_xlabel(r"$\tau\;[-]$", fontsize=14)
    ax_grad.set_ylabel(norm_label_grad, fontsize=14)
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


# =========================================================================================================================================
# All in one plots functions ==============================================================================================================
# =========================================================================================================================================

# Helper method to add the slope triangle in the all-in-one plots _________________________________________________________________________
def _add_slope_triangle(ax, xs, errs, slope, color, tri_size=0.15):

    Cx = np.log10(xs[-2])
    Cy = np.log10(errs[-2])

    Bx, By = Cx, Cy - slope * tri_size
    Ax, Ay = Bx - tri_size, By


    def p(lx, ly):
        return 10**lx, 10**ly

    ax.plot([p(Ax,Ay)[0], p(Bx,By)[0]], [p(Ax,Ay)[1], p(Bx,By)[1]],
            "-", color=color, lw=1.2, alpha=0.8)   # horizontal leg A→B
    ax.plot([p(Cx,Cy)[0], p(Ax,Ay)[0]], [p(Cx,Cy)[1], p(Ax,Ay)[1]],
            "-", color=color, lw=1.2, alpha=0.8)   # vertical leg   C→A
    ax.plot([p(Bx,By)[0], p(Cx,Cy)[0]], [p(Bx,By)[1], p(Cx,Cy)[1]],
            "-", color=color, lw=1.2, alpha=0.8)   # hypotenuse     B→C

    mx = 10 ** (Ax - 0.05 * tri_size)
    my = 10 ** ((Ay + Cy) / 2)
    ax.text(mx, my, str(slope), color=color, fontsize=9,
            va="center", ha="right", alpha=0.9)
    

# Helper method to plot the slope triangle for time plots _________________________________________________________________________________
def _add_slope_triangle_time(ax, dts, errs, slope, color):
    _add_slope_triangle(ax, dts, errs, slope=slope, color=color)

# Function to plot space convergence with all polynomial degrees __________________________________________________________________________
def plot_spatial_convergence_all(hs, errs_c, errs_grad, method=SpaceMethod.LDG, save=False):
    """
    Plots spatial convergence for all polynomial degrees together.

    Parameters
    ----------
    hs        : array-like of mesh sizes (shared across all degrees)
    errs_c    : dict {l: [e0, e1, ...]}  — primal-variable errors per degree
    errs_grad : dict {l: [e0, e1, ...]}  — gradient errors per degree
    method    : SpaceMethod used to pick axis / legend labels
    save      : if True, save the figure to Plots/
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[method]
    norm_label_c, norm_label_grad = NORM_LABELS[method]

    hs      = np.array(hs)
    degrees = sorted(set(errs_c) | set(errs_grad))

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    for l in degrees:
        col = POLY_COLORS.get(l, "black")

        if l in errs_c:
            Ec = np.array(errs_c[l])
            ax_c.loglog(hs, Ec, "o-", color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
            _add_slope_triangle(ax_c, hs, Ec, slope=l + 1, color=col)

        if l in errs_grad:
            Eg = np.array(errs_grad[l])
            ax_grad.loglog(hs, Eg, "s--", color=col, linewidth=2, markersize=7, label=fr"$\ell = {l}$")
            _add_slope_triangle(ax_grad, hs, Eg, slope=l, color=col)

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]

    ax_c.set_title(fr"Errors ${label_c}$",    fontsize=16)
    ax_c.set_xlabel(r"$h\;[-]$",              fontsize=14)
    ax_c.set_ylabel(norm_label_c,             fontsize=14)
    ax_c.set_xlim(hs.min(),        hs.max())
    ax_c.set_ylim(min(all_ec)*0.5, max(all_ec)*2.0)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12)

    ax_grad.set_title(fr"Errors ${label_grad}$",    fontsize=16)
    ax_grad.set_xlabel(r"$h\;[-]$",                 fontsize=14)
    ax_grad.set_ylabel(norm_label_grad,             fontsize=14)
    ax_grad.set_xlim(hs.min(),           hs.max())
    ax_grad.set_ylim(min(all_egrad)*0.5, max(all_egrad)*2.0)
    ax_grad.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_grad.legend(fontsize=12)

    plt.tight_layout()

    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(plot_dir / f"space_convergence_all_{timestamp}.png", dpi=300, bbox_inches="tight")
        print("Space convergence plot (all degrees) saved into Plots folder!")

    plt.show()


# Function to plot all the time degree independently from the method ______________________________________________________________________
def plot_time_convergence_all(dt_list, errs_c, errs_grad, method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False):
    """
    Plots time convergence for all BDF orders or all Theta values together.

    Parameters
    ----------
    dt_list      : array-like of time-step sizes (shared across all orders/thetas)
    errs_c       : dict {order_or_theta: [e0, e1, ...]}  — primal-variable errors
    errs_grad    : dict {order_or_theta: [e0, e1, ...]}  — gradient errors
    method       : TimeMethod.BDF or TimeMethod.THETA
    space_method : SpaceMethod used to pick axis / legend labels
    save         : if True, save the figure to Plots/
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    dt_arr    = np.array(dt_list, dtype=float)
    color_map = TIME_COLORS[method]
    is_bdf    = (method == TimeMethod.BDF)
    keys      = sorted(set(errs_c) | set(errs_grad), key=lambda x: float(x))

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    for key in keys:
        col   = color_map.get(key, "black")
        label = fr"BDF{key}" if is_bdf else fr"$\theta = {key}$"
        expected_order = int(key) if is_bdf else (2 if abs(key - 0.5) < 1e-10 else 1)

        if key in errs_c:
            Ec = np.array(errs_c[key], dtype=float)
            ax_c.loglog(dt_arr, Ec, "o-", color=col, linewidth=2, markersize=7, label=label)
            _add_slope_triangle_time(ax_c, dt_arr, Ec, slope=expected_order, color=col)

        if key in errs_grad:
            Eg = np.array(errs_grad[key], dtype=float)
            ax_grad.loglog(dt_arr, Eg, "s--", color=col, linewidth=2, markersize=7, label=label)
            _add_slope_triangle_time(ax_grad, dt_arr, Eg, slope=expected_order, color=col)

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]
    method_tag = "BDF" if is_bdf else "Theta"

    ax_c.set_title(fr"Errors ${label_c}$ — {method_tag}", fontsize=16)
    ax_c.set_xlabel(r"$\tau\;[-]$",                       fontsize=14)
    ax_c.set_ylabel(norm_label_c,                         fontsize=14)
    ax_c.set_xlim(dt_arr.min(),    dt_arr.max())
    ax_c.set_ylim(min(all_ec)*0.5, max(all_ec)*2.0)
    ax_c.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_c.legend(fontsize=12, loc="upper left")

    ax_grad.set_title(fr"Errors ${label_grad}$ — {method_tag}", fontsize=16)
    ax_grad.set_xlabel(r"$\tau\;[-]$",                          fontsize=14)
    ax_grad.set_ylabel(norm_label_grad,                         fontsize=14)
    ax_grad.set_xlim(dt_arr.min(),       dt_arr.max())
    ax_grad.set_ylim(min(all_egrad)*0.5, max(all_egrad)*2.0)
    ax_grad.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_grad.legend(fontsize=12, loc="upper left")

    plt.tight_layout()

    if save:
        plot_dir = Path(__file__).parent / "Plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(plot_dir / f"time_convergence_{method_tag}_all_{timestamp}.png", dpi=300, bbox_inches="tight")
        print(f"Time convergence plot (all {method_tag}) saved into Plots folder!")

    plt.show()