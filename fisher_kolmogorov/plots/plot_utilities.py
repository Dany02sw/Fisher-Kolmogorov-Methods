import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from fisher_kolmogorov.config                         import get_convergence_dir
from fisher_kolmogorov.utilities.enum_utilities       import SpaceMethod, TimeMethod, BdfOrder, PolyDegree, ThetaMethod
from fisher_kolmogorov.utilities.dictionary_utilities import ERROR_LABELS_PLOT, NORM_LABELS

from fisher_kolmogorov.plots._primitives import save_plot
from fisher_kolmogorov.plots._axes       import finalize_ax, finalize_ax_pair
from fisher_kolmogorov.plots._curves     import (
    plot_spatial_curves,
    plot_poly_curves,
    plot_time_curves,
    plot_saturation_curves_loglog,
    plot_saturation_curves_semilogy,
    _time_label,
    _expected_time_order,
)


# =========================================================================================================================================
# Color maps ==============================================================================================================================
# =========================================================================================================================================

POLY_COLORS = {
    PolyDegree.P1: "red",
    PolyDegree.P2: "darkmagenta",
    PolyDegree.P3: "darkcyan",
    PolyDegree.P4: "springgreen",
    PolyDegree.P5: "blue",
    PolyDegree.P6: "crimson",
    PolyDegree.P7: "gold",
    PolyDegree.P8: "magenta",
}  # One color per polynomial degree, used consistently across all convergence plots

BDF_COLORS = {
    BdfOrder.BDF1: "yellowgreen",
    BdfOrder.BDF2: "deepskyblue",
    BdfOrder.BDF3: "blue",
    BdfOrder.BDF4: "deeppink",
    BdfOrder.BDF5: "crimson",
    BdfOrder.BDF6: "darkorange",
}

THETA_COLORS = {
    ThetaMethod.EE: "gold",
    ThetaMethod.CN: "deeppink",
    ThetaMethod.IE: "cornflowerblue",
}

TIME_COLORS = {
    TimeMethod.BDF:   BDF_COLORS,
    TimeMethod.THETA: THETA_COLORS,
}


# =========================================================================================================================================
# Single-order / single-degree plots ======================================================================================================
# =========================================================================================================================================

def plot_spatial_convergence(hs, err_c, err_grad, l: PolyDegree,
                             space_method=SpaceMethod.LDG, time_method=TimeMethod.BDF, save=False):
    """
    Plot spatial convergence for a single polynomial degree.

    Parameters
    ----------
    hs       : array-like — mesh sizes
    err_c    : array-like — primal-variable errors
    err_grad : array-like — gradient errors
    l        : PolyDegree
    method   : SpaceMethod
    save     : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    hs  = np.array(hs, dtype=float)

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    # Single-degree: one curve + ref line (no triangle, no legend clutter)
    plot_spatial_curves(
        ax_c, ax_grad,
        hs,
        errs_c    = {l: err_c},
        errs_grad = {l: err_grad},
        poly_colors  = POLY_COLORS,
        use_triangles = False,
    )

    all_ec    = np.array(err_c,    dtype=float)
    all_egrad = np.array(err_grad, dtype=float)

    ax_c.set_xlim(hs.min(),    hs.max())
    ax_grad.set_xlim(hs.min(), hs.max())

    finalize_ax_pair(
        ax_c, ax_grad,
        all_ec, all_egrad,
        norm_label_c,    norm_label_grad,
        fr"Errors ${label_c}$", fr"Errors ${label_grad}$",
        xlabel     = r"$h\;[-]$",
        legend_loc = "best",
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_space_l{l}_{timestamp}.png",
            f"Space convergence plot for l={int(l)}",
        )

    plt.show()


def plot_polynomial_convergence(errors_c, errors_grad, l_list, h,
                                space_method=SpaceMethod.LDG, time_method=TimeMethod.BDF, save=False):
    """
    Plot polynomial convergence for a single fixed mesh size.

    Parameters
    ----------
    errors_c    : array-like — primal errors per degree
    errors_grad : array-like — gradient errors per degree
    l_list      : list of PolyDegree
    h           : float — fixed mesh size
    method      : SpaceMethod
    save        : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    l_ints = [int(l) for l in l_list]

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_poly_curves(ax, l_ints, errors_c, errors_grad, label_c, label_grad, h)

    all_e = np.concatenate([np.array(errors_c, dtype=float),
                            np.array(errors_grad, dtype=float)])
    finalize_ax(
        ax, all_e,
        norm_label  = f"{norm_label_c}  /  {norm_label_grad}",
        title       = fr"Errors ${label_c}$ and ${label_grad}$",
        xlabel      = r"$\ell$",
        legend_loc  = "upper right",
        fontsize_title  = 20,
        fontsize_labels = 16,
        fontsize_legend = 14,
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_polynomial_{timestamp}.png",
            "Polynomial convergence plot",
        )

    plt.show()


def plot_time_convergence(dt_list, err_c, err_grad, order,
                          time_method=TimeMethod.BDF, space_method=SpaceMethod.LDG,
                          save=False):
    """
    Plot time convergence for a single BDF order or Theta value.

    Parameters
    ----------
    dt_list      : array-like — time-step sizes
    err_c        : array-like — primal errors
    err_grad     : array-like — gradient errors
    order        : BdfOrder or ThetaMethod
    method       : TimeMethod
    space_method : SpaceMethod
    save         : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    is_bdf  = (time_method == TimeMethod.BDF)
    label   = _time_label(order, is_bdf)
    dt_arr  = np.array(dt_list, dtype=float)

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    plot_time_curves(
        ax_c, ax_grad, dt_arr,
        errs_c    = {order: err_c},
        errs_grad = {order: err_grad},
        time_method  = time_method,
        color_map    = TIME_COLORS[time_method],
        use_triangles = False,
    )

    all_ec    = np.array(err_c,    dtype=float)
    all_egrad = np.array(err_grad, dtype=float)

    ax_c.set_xlim(dt_arr.min(),    dt_arr.max())
    ax_grad.set_xlim(dt_arr.min(), dt_arr.max())

    finalize_ax_pair(
        ax_c, ax_grad,
        all_ec, all_egrad,
        norm_label_c,    norm_label_grad,
        fr"Errors ${label_c}$ — {label}", fr"Errors ${label_grad}$ — {label}",
        xlabel     = r"$\tau\;[-]$",
        legend_loc = "upper left",
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix    = f"BDF{int(order)}" if is_bdf else f"THETA{float(order)}"
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_time_{suffix}_{timestamp}.png",
            "Time convergence plot",
        )

    plt.show()


# =========================================================================================================================================
# All-in-one plots ========================================================================================================================
# =========================================================================================================================================

def plot_spatial_convergence_all(hs, errs_c, errs_grad,
                                 space_method=SpaceMethod.LDG,
                                 time_method=TimeMethod.BDF, save=False):
    """
    Plot spatial convergence for all polynomial degrees together.

    Parameters
    ----------
    hs        : array-like — mesh sizes (shared across all degrees)
    errs_c    : dict {PolyDegree: [e0, ...]} — primal errors
    errs_grad : dict {PolyDegree: [e0, ...]} — gradient errors
    space_method : SpaceMethod
    time_method  : TimeMethod
    save         : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    hs = np.array(hs, dtype=float)

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    plot_spatial_curves(ax_c, ax_grad, hs, errs_c, errs_grad,
                        POLY_COLORS, use_triangles=True)

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]

    ax_c.set_xlim(hs.min(),    hs.max())
    ax_grad.set_xlim(hs.min(), hs.max())

    finalize_ax_pair(
        ax_c, ax_grad,
        all_ec, all_egrad,
        norm_label_c,    norm_label_grad,
        fr"Errors ${label_c}$", fr"Errors ${label_grad}$",
        xlabel     = r"$h\;[-]$",
        legend_loc = "lower right",
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_space_all_{timestamp}.png",
            "Space convergence plot (all degrees)",
        )

    plt.show()


def plot_time_convergence_all(dt_list, errs_c, errs_grad,
                              time_method=TimeMethod.BDF,
                              space_method=SpaceMethod.LDG, save=False):
    """
    Plot time convergence for all BDF orders or all Theta values together.

    Parameters
    ----------
    dt_list      : array-like — time-step sizes (shared across all orders)
    errs_c       : dict {BdfOrder | ThetaMethod: [e0, ...]}
    errs_grad    : dict {BdfOrder | ThetaMethod: [e0, ...]}
    time_method  : TimeMethod
    space_method : SpaceMethod
    save         : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    dt_arr     = np.array(dt_list, dtype=float)
    is_bdf     = (time_method == TimeMethod.BDF)
    method_tag = "BDF" if is_bdf else "Theta"

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    plot_time_curves(ax_c, ax_grad, dt_arr, errs_c, errs_grad,
                     time_method, TIME_COLORS[time_method], use_triangles=True)

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]

    ax_c.set_xlim(dt_arr.min(),    dt_arr.max())
    ax_grad.set_xlim(dt_arr.min(), dt_arr.max())

    finalize_ax_pair(
        ax_c, ax_grad,
        all_ec, all_egrad,
        norm_label_c,    norm_label_grad,
        fr"Errors ${label_c}$ — {method_tag}", fr"Errors ${label_grad}$ — {method_tag}",
        xlabel     = r"$\tau\;[-]$",
        legend_loc = "lower right",
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_time_all_{timestamp}.png",
            f"Time convergence plot (all {method_tag})",
        )

    plt.show()


def plot_combined_poly_and_time(
    errors_c_poly, errors_grad_poly, l_list, h,
    dt_list, errs_c_time,
    time_method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False,
):
    """
    Side-by-side figure: polynomial convergence (left) + temporal convergence (right).

    Parameters
    ----------
    errors_c_poly    : array-like — primal errors per degree
    errors_grad_poly : array-like — gradient errors per degree
    l_list           : list of PolyDegree
    h                : float — fixed mesh size for polynomial study
    dt_list          : array-like — time-step sizes
    errs_c_time      : dict {BdfOrder | ThetaMethod: [e0, ...]}
    time_method      : TimeMethod
    space_method     : SpaceMethod
    save             : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    l_ints     = [int(l) for l in l_list]
    dt_arr     = np.array(dt_list, dtype=float)
    is_bdf     = (time_method == TimeMethod.BDF)
    method_tag = "BDF" if is_bdf else "Theta"

    fig, (ax_poly, ax_time) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: polynomial convergence
    plot_poly_curves(ax_poly, l_ints, errors_c_poly, errors_grad_poly,
                     label_c, label_grad, h)

    all_poly = np.concatenate([np.array(errors_c_poly,    dtype=float),
                               np.array(errors_grad_poly, dtype=float)])
    finalize_ax(
        ax_poly, all_poly,
        norm_label  = f"{norm_label_c}  /  {norm_label_grad}",
        title       = fr"Errors ${label_c}$ and ${label_grad}$",
        xlabel      = r"$\ell$",
        legend_loc  = "upper right",
        fontsize_title  = 16,
        fontsize_labels = 16,
        fontsize_legend = 12,
    )

    # Right: temporal convergence (primal only — no ax_grad panel here)
    plot_time_curves(ax_time, None, dt_arr,
                     errs_c       = errs_c_time,
                     errs_grad    = {},
                     time_method  = time_method,
                     color_map    = TIME_COLORS[time_method],
                     use_triangles = True)

    all_ec_time = [v for vals in errs_c_time.values() for v in vals]
    ax_time.set_xlim(dt_arr.min(), dt_arr.max())
    finalize_ax(
        ax_time, np.array(all_ec_time, dtype=float),
        norm_label  = norm_label_c,
        title       = fr"Errors ${label_c}$ — {method_tag}",
        xlabel      = r"$\tau\;[-]$",
        legend_loc  = "lower right",
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_combined_{timestamp}.png",
            f"Combined polynomial + time convergence plot ({method_tag})",
        )

    plt.show()


# =========================================================================================================================================
# Saturation study ========================================================================================================================
# =========================================================================================================================================

def plot_spatial_saturation(hs, errs_c, errs_grad, l: PolyDegree,
                            time_method=TimeMethod.BDF,
                            space_method=SpaceMethod.LDG, save=False):
    """
    Plot spatial saturation: fixed polynomial degree, varying time order.

    Parameters
    ----------
    hs           : array-like — mesh sizes
    errs_c       : dict {BdfOrder | ThetaMethod: [e0, ...]}
    errs_grad    : dict {BdfOrder | ThetaMethod: [e0, ...]}
    l            : PolyDegree — fixed polynomial degree
    time_method  : TimeMethod
    space_method : SpaceMethod
    save         : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    hs         = np.array(hs, dtype=float)
    is_bdf     = (time_method == TimeMethod.BDF)
    method_tag = "BDF" if is_bdf else "Theta"

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    plot_saturation_curves_loglog(
        ax_c, ax_grad, hs, errs_c, errs_grad,
        time_method  = time_method,
        color_map    = TIME_COLORS[time_method],
        ref_slope_c  = int(l) + 1,
        ref_slope_grad = int(l),
        x_label      = "h",
    )

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]

    ax_c.set_xlim(hs.min(),    hs.max())
    ax_grad.set_xlim(hs.min(), hs.max())

    finalize_ax_pair(
        ax_c, ax_grad,
        all_ec, all_egrad,
        norm_label_c,    norm_label_grad,
        fr"Errors ${label_c}$  —  $\ell = {int(l)}$  —  {method_tag} saturation",
        fr"Errors ${label_grad}$  —  $\ell = {int(l)}$  —  {method_tag} saturation",
        xlabel     = r"$h\;[-]$",
        legend_loc = "best",
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_sat_space_l{int(l)}_{timestamp}.png",
            f"Space saturation plot (l={int(l)}, {method_tag})",
        )

    plt.show()


def plot_polynomial_saturation(l_list, errs_c, errs_grad, h: float,
                               time_method=TimeMethod.BDF,
                               space_method=SpaceMethod.LDG, save=False):
    """
    Plot polynomial saturation: fixed mesh size, varying time order.

    Parameters
    ----------
    l_list       : list of PolyDegree
    errs_c       : dict {BdfOrder | ThetaMethod: [e0, ...]}
    errs_grad    : dict {BdfOrder | ThetaMethod: [e0, ...]}
    h            : float — fixed mesh size
    time_method  : TimeMethod
    space_method : SpaceMethod
    save         : bool
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    l_ints     = [int(l) for l in l_list]
    is_bdf     = (time_method == TimeMethod.BDF)
    method_tag = "BDF" if is_bdf else "Theta"

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    plot_saturation_curves_semilogy(
        ax_c, ax_grad, l_ints, errs_c, errs_grad,
        time_method, TIME_COLORS[time_method], h,
    )

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]

    for ax in (ax_c, ax_grad):
        ax.set_xlim(l_ints[0], l_ints[-1])
        ax.set_xticks(np.arange(l_ints[0], l_ints[-1] + 1))

    finalize_ax_pair(
        ax_c, ax_grad,
        all_ec, all_egrad,
        norm_label_c,    norm_label_grad,
        fr"Errors ${label_c}$  —  $h = {h}$  —  {method_tag} saturation",
        fr"Errors ${label_grad}$  —  $h = {h}$  —  {method_tag} saturation",
        xlabel     = r"$\ell$",
        legend_loc = "best",
    )

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_sat_poly_h{h}_{timestamp}.png",
            f"Polynomial saturation plot ({method_tag}, h={h})",
        )

    plt.show()


def plot_spatial_saturation_combined(hs, errs_c_by_degree, errs_grad_by_degree,
                                     degrees, time_method=TimeMethod.BDF,
                                     space_method=SpaceMethod.LDG, save=False):
    """
    Plot a 2×2 spatial saturation study for two polynomial degrees.

    Parameters
    ----------
    hs                  : array-like — mesh sizes (shared)
    errs_c_by_degree    : dict {PolyDegree: {BdfOrder | ThetaMethod: [e0, ...]}}
    errs_grad_by_degree : dict {PolyDegree: {BdfOrder | ThetaMethod: [e0, ...]}}
    degrees             : list of PolyDegree — exactly two entries, top row first
    time_method         : TimeMethod
    space_method        : SpaceMethod
    save                : bool
    """
    if len(degrees) != 2:
        raise ValueError("degrees must contain exactly two PolyDegree entries.")

    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    hs         = np.array(hs, dtype=float)
    is_bdf     = (time_method == TimeMethod.BDF)
    method_tag = "BDF" if is_bdf else "Theta"

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for row, l in enumerate(degrees):
        ax_c, ax_grad = axes[row, 0], axes[row, 1]
        errs_c    = errs_c_by_degree[l]
        errs_grad = errs_grad_by_degree[l]

        plot_saturation_curves_loglog(
            ax_c, ax_grad, hs, errs_c, errs_grad,
            time_method    = time_method,
            color_map      = TIME_COLORS[time_method],
            ref_slope_c    = int(l) + 1,
            ref_slope_grad = int(l),
            x_label        = "h",
        )

        all_ec    = [v for vals in errs_c.values()    for v in vals]
        all_egrad = [v for vals in errs_grad.values() for v in vals]

        ax_c.set_xlim(hs.min(),    hs.max())
        ax_grad.set_xlim(hs.min(), hs.max())

        finalize_ax_pair(
            ax_c, ax_grad,
            all_ec, all_egrad,
            norm_label_c,    norm_label_grad,
            fr"Errors ${label_c}$  —  $\ell = {int(l)}$  —  {method_tag} saturation",
            fr"Errors ${label_grad}$  —  $\ell = {int(l)}$  —  {method_tag} saturation",
            xlabel          = r"$h\;[-]$",
            legend_loc      = "best",
            fontsize_title  = 15,
            fontsize_labels = 13,
            fontsize_legend = 11,
        )

    plt.tight_layout()

    if save:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
        deg_tag    = "_".join(f"l{int(d)}" for d in degrees)
        save_plot(
            get_convergence_dir(space_method, time_method)
            / f"{space_method.name}{time_method.name}_sat_space_{deg_tag}_{timestamp}.png",
            f"Space saturation combined plot ({deg_tag}, {method_tag})",
        )

    plt.show()