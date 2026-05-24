import matplotlib.pyplot as plt
import numpy             as np

from pathlib  import Path
from datetime import datetime

from Utilities.EnumUtilities       import SpaceMethod, TimeMethod, BdfOrder, PolyDegree, ThetaMethod
from Utilities.DictionaryUtilities import ERROR_LABELS_PLOT, NORM_LABELS


# =========================================================================================================================================
# Color maps ==============================================================================================================================
# =========================================================================================================================================

POLY_COLORS = {
    PolyDegree.P1: "red",
    PolyDegree.P2: "darkmagenta",
    PolyDegree.P3: "darkcyan",
    PolyDegree.P4: "yellowgreen",
    PolyDegree.P5: "blue",
    PolyDegree.P6: "crimson",
    PolyDegree.P7: "gold",
    PolyDegree.P8: "lime",
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
# Private helpers =========================================================================================================================
# =========================================================================================================================================

def _add_ref_line(ax, xs, e0, slope, x_label):
    """
    Draws a black dashed reference line anchored at the first point.

    The line follows  e(x) = e0 * (xs / xs[0])^slope  and is labeled
    according to x_label ('h' for spatial, 'tau' for temporal plots).

    Parameters
    ----------
    ax      : matplotlib Axes
    xs      : array-like of x-axis values (mesh sizes or time steps)
    e0      : error value at xs[0], used as anchor
    slope   : integer or float exponent for the reference line
    x_label : str, either 'h' or 'tau' — selects the LaTeX label symbol
    """
    xs  = np.asarray(xs, dtype=float)
    sym = r"\tau" if x_label == "tau" else "h"
    ref = e0 * (xs / xs[0]) ** slope
    ax.loglog(xs, ref, "k--", linewidth=1.5, alpha=0.6,
              label=fr"$O({sym}^{{{slope}}})$")

def _add_slope_triangle(ax, xs, errs, slope, color, tri_frac=0.18):
    """
    Draws a small slope triangle near the last two points of a loglog curve.
 
    The triangle size scales with the log10 range of xs so that it looks
    consistent regardless of the x-axis span.
 
    Parameters
    ----------
    ax       : matplotlib Axes
    xs       : array-like of x-axis values
    errs     : array-like of error values
    slope    : integer exponent to annotate
    color    : triangle and label color
    tri_frac : triangle leg size as a fraction of the log10 x-range
    """
    xs   = np.asarray(xs,   dtype=float)
    errs = np.asarray(errs, dtype=float)
 
    x_span   = abs(np.log10(xs.max()) - np.log10(xs.min()))
    tri_size = tri_frac * x_span
 
    Cx = np.log10(xs[-2])
    Cy = np.log10(errs[-2]) - tri_size
 
    Bx, By = Cx, Cy - slope * tri_size
    Ax, Ay = Bx - tri_size, By
 
    def p(lx, ly):
        return 10**lx, 10**ly
 
    ax.plot([p(Ax, Ay)[0], p(Bx, By)[0]], [p(Ax, Ay)[1], p(Bx, By)[1]],
            "-", color=color, lw=1.2, alpha=0.8)
    ax.plot([p(Cx, Cy)[0], p(Ax, Ay)[0]], [p(Cx, Cy)[1], p(Ax, Ay)[1]],
            "-", color=color, lw=1.2, alpha=0.8)
    ax.plot([p(Bx, By)[0], p(Cx, Cy)[0]], [p(Bx, By)[1], p(Cx, Cy)[1]],
            "-", color=color, lw=1.2, alpha=0.8)
 
    mx = 10 ** (Cx + 0.10 * tri_size)
    my = 10 ** ((Cy + By) / 2)
    ax.text(mx, my, str(slope), color=color, fontsize=9,
            va="center", ha="left", alpha=0.9)

def _save_plot(fig_path: Path, label: str):
    """Saves the current figure and prints a confirmation message."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"{label} saved into Plots folder!")


# =========================================================================================================================================
# Single-order / single-degree plots ======================================================================================================
# =========================================================================================================================================

def plot_spatial_convergence(hs, err_c, err_grad, l: PolyDegree, method=SpaceMethod.LDG, save=False):
    """
    Plots spatial convergence for a single polynomial degree.

    Parameters
    ----------
    hs       : array-like of mesh sizes
    err_c    : array-like of primal-variable errors
    err_grad : array-like of gradient errors
    l        : PolyDegree — polynomial degree used
    method   : SpaceMethod used to pick axis / legend labels
    save     : if True, save the figure to Plots/
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[method]
    norm_label_c, norm_label_grad = NORM_LABELS[method]

    hs    = np.array(hs,       dtype=float)
    Ec    = np.array(err_c,    dtype=float)
    Egrad = np.array(err_grad, dtype=float)
    col   = POLY_COLORS.get(l, "black")

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    ax_c.loglog(hs, Ec,    "o-",  color=col, linewidth=2, markersize=7, label=fr"$\ell = {int(l)}$")
    _add_ref_line(ax_c, hs, Ec[0], slope=int(l) + 1, x_label="h")

    ax_grad.loglog(hs, Egrad, "s--", color=col, linewidth=2, markersize=7, label=fr"$\ell = {int(l)}$")
    _add_ref_line(ax_grad, hs, Egrad[0], slope=int(l), x_label="h")

    for ax, E, norm_label, title_label in [
        (ax_c,    Ec,    norm_label_c,    label_c),
        (ax_grad, Egrad, norm_label_grad, label_grad),
    ]:
        ax.set_title(fr"Errors ${title_label}$", fontsize=16)
        ax.set_xlabel(r"$h\;[-]$",               fontsize=14)
        ax.set_ylabel(norm_label,                fontsize=14)
        ax.grid(True, which="both", linestyle=":", alpha=0.7)
        ax.legend(fontsize=12)
        ax.set_xlim(hs.min(),    hs.max())
        ax.set_ylim(E.min()*0.5, E.max()*2.0)

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            Path(__file__).parent / "Plots" / f"space_convergence_l{int(l)}_{timestamp}.png",
            f"Space convergence plot for l={int(l)}",
        )

    plt.show()

def plot_polynomial_convergence(errors_c, errors_grad, l_list, h, method=SpaceMethod.LDG, save=False):
    """
    Plots polynomial convergence of both a base variable and a gradient-based one.

    Parameters
    ----------
    errors_c    : array-like of primal-variable errors per degree
    errors_grad : array-like of gradient errors per degree
    l_list      : list of PolyDegree values tested
    h           : float, fixed mesh size used
    method      : SpaceMethod used to pick axis / legend labels
    save        : if True, save the figure to Plots/
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[method]
    norm_label_c, norm_label_grad = NORM_LABELS[method]

    l_ints   = [int(l) for l in l_list]
    L_ext    = np.arange(l_ints[0], l_ints[-1] + 1)
    L_num    = np.array(l_ints)
    Ec_vals  = np.array(errors_c)
    Eg_vals  = np.array(errors_grad)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(L_num, Ec_vals, "-o",  color="yellowgreen", linewidth=2, markersize=8, label=fr"${label_c}$")
    ax.semilogy(L_num, Eg_vals, "--s", color="lime",        linewidth=2, markersize=8, label=fr"${label_grad}$")

    ref = Ec_vals[0] * (h ** (L_ext - l_ints[0]))
    ax.semilogy(L_ext, ref, "k--", linewidth=3, label=r"$h^{\ell}$")

    ax.set_xlim(l_ints[0], l_ints[-1])
    ax.set_xticks(np.arange(l_ints[0], l_ints[-1] + 1))
    ax.set_xlabel(r"$\ell$",                                    fontsize=16)
    ax.set_ylabel(f"{norm_label_c}  /  {norm_label_grad}",     fontsize=14)
    ax.set_title(fr"Errors ${label_c}$ and ${label_grad}$",    fontsize=20)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(fontsize=14, loc="upper right")
    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            Path(__file__).parent / "Plots" / f"polynomial_convergence_{timestamp}.png",
            "Polynomial convergence plot",
        )

    plt.show()

def plot_time_convergence(dt_list, err_c, err_grad, order, method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False):
    """
    Plots time convergence for a single BDF order or Theta value.

    Parameters
    ----------
    dt_list      : array-like of time-step sizes
    err_c        : array-like of primal-variable errors
    err_grad     : array-like of gradient errors
    order        : BdfOrder or ThetaMethod — time integration parameter
    method       : TimeMethod.BDF or TimeMethod.THETA
    space_method : SpaceMethod used to pick axis / legend labels
    save         : if True, save the figure to Plots/
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    dt_arr = np.array(dt_list,  dtype=float)
    Ec     = np.array(err_c,    dtype=float)
    Egrad  = np.array(err_grad, dtype=float)

    is_bdf         = (method == TimeMethod.BDF)
    col            = TIME_COLORS[method].get(order, "black")
    label          = fr"BDF{int(order)}" if is_bdf else fr"$\theta = {float(order)}$"
    expected_order = int(order) if is_bdf else (2 if order == ThetaMethod.CN else 1)

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    ax_c.loglog(dt_arr, Ec,    "o-",  color=col, linewidth=2, markersize=7, label=label)
    _add_ref_line(ax_c, dt_arr, Ec[0], slope=expected_order, x_label="tau")

    ax_grad.loglog(dt_arr, Egrad, "s--", color=col, linewidth=2, markersize=7, label=label)
    _add_ref_line(ax_grad, dt_arr, Egrad[0], slope=expected_order, x_label="tau")

    for ax, E, norm_label, title_label in [
        (ax_c,    Ec,    norm_label_c,    label_c),
        (ax_grad, Egrad, norm_label_grad, label_grad),
    ]:
        ax.set_title(fr"Errors ${title_label}$ — {label}", fontsize=16)
        ax.set_xlabel(r"$\tau\;[-]$",                      fontsize=14)
        ax.set_ylabel(norm_label,                          fontsize=14)
        ax.grid(True, which="both", linestyle=":", alpha=0.7)
        ax.legend(fontsize=12, loc="upper left")
        ax.set_xlim(dt_arr.min(), dt_arr.max())
        ax.set_ylim(E.min()*0.5,  E.max()*2.0)

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = f"BDF{int(order)}" if is_bdf else f"THETA{float(order)}"
        _save_plot(
            Path(__file__).parent / "Plots" / f"time_convergence_{suffix}_{timestamp}.png",
            "Time convergence plot",
        )

    plt.show()


# =========================================================================================================================================
# All-in-one plots ========================================================================================================================
# =========================================================================================================================================

def plot_spatial_convergence_all(hs, errs_c, errs_grad, method=SpaceMethod.LDG, save=False):
    """
    Plots spatial convergence for all polynomial degrees together.

    Parameters
    ----------
    hs        : array-like of mesh sizes (shared across all degrees)
    errs_c    : dict {PolyDegree: [e0, e1, ...]}  — primal-variable errors per degree
    errs_grad : dict {PolyDegree: [e0, e1, ...]}  — gradient errors per degree
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
            ax_c.loglog(hs, Ec, "o-", color=col, linewidth=2, markersize=7, label=fr"$\ell = {int(l)}$")
            _add_slope_triangle(ax_c, hs, Ec, slope=int(l) + 1, color=col)

        if l in errs_grad:
            Eg = np.array(errs_grad[l])
            ax_grad.loglog(hs, Eg, "s--", color=col, linewidth=2, markersize=7, label=fr"$\ell = {int(l)}$")
            _add_slope_triangle(ax_grad, hs, Eg, slope=int(l), color=col)

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]

    for ax, all_e, norm_label, title_label in [
        (ax_c,    all_ec,    norm_label_c,    label_c),
        (ax_grad, all_egrad, norm_label_grad, label_grad),
    ]:
        ax.set_title(fr"Errors ${title_label}$", fontsize=16)
        ax.set_xlabel(r"$h\;[-]$",               fontsize=14)
        ax.set_ylabel(norm_label,                fontsize=14)
        ax.set_xlim(hs.min(),        hs.max())
        ax.set_ylim(min(all_e)*0.5,  max(all_e)*2.0)
        ax.grid(True, which="both", linestyle=":", alpha=0.7)
        ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            Path(__file__).parent / "Plots" / f"space_convergence_all_{timestamp}.png",
            "Space convergence plot (all degrees)",
        )

    plt.show()

def plot_time_convergence_all(dt_list, errs_c, errs_grad, method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False):
    """
    Plots time convergence for all BDF orders or all Theta values together.

    Parameters
    ----------
    dt_list      : array-like of time-step sizes (shared across all orders/thetas)
    errs_c       : dict {BdfOrder | ThetaMethod: [e0, e1, ...]}  — primal-variable errors
    errs_grad    : dict {BdfOrder | ThetaMethod: [e0, e1, ...]}  — gradient errors
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
        label = fr"BDF{int(key)}" if is_bdf else fr"$\theta = {float(key)}$"
        expected_order = int(key) if is_bdf else (2 if key == ThetaMethod.CN else 1)

        if key in errs_c:
            Ec = np.array(errs_c[key], dtype=float)
            ax_c.loglog(dt_arr, Ec, "o-", color=col, linewidth=2, markersize=7, label=label)
            _add_slope_triangle(ax_c, dt_arr, Ec, slope=expected_order, color=col)

        if key in errs_grad:
            Eg = np.array(errs_grad[key], dtype=float)
            ax_grad.loglog(dt_arr, Eg, "s--", color=col, linewidth=2, markersize=7, label=label)
            _add_slope_triangle(ax_grad, dt_arr, Eg, slope=expected_order, color=col)

    all_ec     = [v for vals in errs_c.values()    for v in vals]
    all_egrad  = [v for vals in errs_grad.values() for v in vals]
    method_tag = "BDF" if is_bdf else "Theta"

    for ax, all_e, norm_label, title_label in [
        (ax_c,    all_ec,    norm_label_c,    label_c),
        (ax_grad, all_egrad, norm_label_grad, label_grad),
    ]:
        ax.set_title(fr"Errors ${title_label}$ — {method_tag}", fontsize=16)
        ax.set_xlabel(r"$\tau\;[-]$",                           fontsize=14)
        ax.set_ylabel(norm_label,                               fontsize=14)
        ax.set_xlim(dt_arr.min(),   dt_arr.max())
        ax.set_ylim(min(all_e)*0.5, max(all_e)*2.0)
        ax.grid(True, which="both", linestyle=":", alpha=0.7)
        ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            Path(__file__).parent / "Plots" / f"time_convergence_{method_tag}_all_{timestamp}.png",
            f"Time convergence plot (all {method_tag})",
        )

    plt.show()

def plot_combined_poly_and_time(
    # Polynomial convergence inputs
    errors_c_poly, errors_grad_poly, l_list, h,
    # Time convergence inputs (primal variable only)
    dt_list, errs_c_time,
    # Common options
    method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False,
):
    """
    Renders a side-by-side figure with:
      - Left  : polynomial convergence of both the primal variable and the gradient.
      - Right : temporal convergence of the primal variable for every BDF order
                or Theta value supplied.

    Parameters
    ----------
    errors_c_poly    : array-like — primal errors, one per degree in l_list
    errors_grad_poly : array-like — gradient errors, one per degree in l_list
    l_list           : list of PolyDegree — polynomial degrees tested
    h                : float — fixed mesh size used for the polynomial study
    dt_list          : array-like — time-step sizes (shared across all orders/thetas)
    errs_c_time      : dict {BdfOrder | ThetaMethod: [e0, e1, ...]} — primal errors
    method           : TimeMethod.BDF or TimeMethod.THETA
    space_method     : SpaceMethod used to pick axis / legend labels
    save             : if True, saves the figure to Plots/
    """
    label_c, label_grad           = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    l_ints  = [int(l) for l in l_list]
    L_num   = np.array(l_ints)
    L_ext   = np.arange(l_ints[0], l_ints[-1] + 1)
    Ec_poly = np.array(errors_c_poly,    dtype=float)
    Eg_poly = np.array(errors_grad_poly, dtype=float)

    dt_arr    = np.array(dt_list, dtype=float)
    color_map = TIME_COLORS[method]
    is_bdf    = (method == TimeMethod.BDF)
    keys      = sorted(errs_c_time.keys(), key=lambda x: float(x))

    fig, (ax_poly, ax_time) = plt.subplots(1, 2, figsize=(15, 6))

    # ── Left: polynomial convergence ─────────────────────────────────────────
    ax_poly.semilogy(L_num, Ec_poly, "-o",  color="yellowgreen", linewidth=2,
                     markersize=8, label=fr"${label_c}$")
    ax_poly.semilogy(L_num, Eg_poly, "--s", color="lime",        linewidth=2,
                     markersize=8, label=fr"${label_grad}$")

    ref = Ec_poly[0] * (h ** (L_ext - l_ints[0]))
    ax_poly.semilogy(L_ext, ref, "k--", linewidth=1.5, alpha=0.6, label=r"$h^{\ell}$")

    ax_poly.set_xlim(l_ints[0], l_ints[-1])
    ax_poly.set_xticks(np.arange(l_ints[0], l_ints[-1] + 1))
    ax_poly.set_xlabel(r"$\ell$",                                  fontsize=16)
    ax_poly.set_ylabel(f"{norm_label_c}  /  {norm_label_grad}",   fontsize=14)
    ax_poly.set_title(fr"Errors ${label_c}$ and ${label_grad}$",  fontsize=16)
    ax_poly.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_poly.legend(fontsize=12, loc="upper right")

    # ── Right: time convergence (primal variable only) ────────────────────────
    for key in keys:
        col   = color_map.get(key, "black")
        label = fr"BDF{int(key)}" if is_bdf else fr"$\theta = {float(key)}$"
        expected_order = int(key) if is_bdf else (2 if key == ThetaMethod.CN else 1)

        Ec = np.array(errs_c_time[key], dtype=float)
        ax_time.loglog(dt_arr, Ec, "o-", color=col, linewidth=2, markersize=7, label=label)
        _add_slope_triangle(ax_time, dt_arr, Ec, slope=expected_order, color=col)

    all_ec_time = [v for vals in errs_c_time.values() for v in vals]
    method_tag  = "BDF" if is_bdf else "Theta"

    ax_time.set_title(fr"Errors ${label_c}$ — {method_tag}", fontsize=16)
    ax_time.set_xlabel(r"$\tau\;[-]$",                       fontsize=14)
    ax_time.set_ylabel(norm_label_c,                         fontsize=14)
    ax_time.set_xlim(dt_arr.min(),       dt_arr.max())
    ax_time.set_ylim(min(all_ec_time) * 0.5, max(all_ec_time) * 2.0)
    ax_time.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_time.legend(fontsize=12, loc="lower right")

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            Path(__file__).parent / "Plots" / f"combined_poly_time_{method_tag}_{timestamp}.png",
            f"Combined polynomial + time convergence plot ({method_tag})",
        )

    plt.show()


# =========================================================================================================================================
# Saturation study ========================================================================================================================
# =========================================================================================================================================

def plot_spatial_saturation(hs, errs_c, errs_grad, l: PolyDegree,
                            method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False):
    """
    Plots a spatial saturation study: fixed polynomial degree, varying time
    integration order (BDF order or Theta value).

    The reference slope line is anchored at the first point of the highest
    BDF order (or the implicit Theta, which saturates latest). Slopes are
    determined by the spatial polynomial degree l: l+1 for the primal
    variable and l for the gradient.

    Parameters
    ----------
    hs           : array-like of mesh sizes (shared across all keys)
    errs_c       : dict {BdfOrder | ThetaMethod: [e0, e1, ...]}  — primal errors
    errs_grad    : dict {BdfOrder | ThetaMethod: [e0, e1, ...]}  — gradient errors
    l            : PolyDegree — fixed polynomial degree used in all runs
    method       : TimeMethod.BDF or TimeMethod.THETA
    space_method : SpaceMethod used to pick axis / legend labels
    save         : if True, save the figure to Plots/
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    hs        = np.array(hs, dtype=float)
    color_map = TIME_COLORS[method]
    is_bdf    = (method == TimeMethod.BDF)
    keys      = sorted(set(errs_c) | set(errs_grad), key=lambda x: float(x))

    # For BDF: highest order saturates last; for Theta: implicit (1.0) saturates last
    key_ref = max(keys, key=lambda x: float(x))

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    for key in keys:
        col   = color_map.get(key, "gray")
        label = fr"BDF{int(key)}" if is_bdf else fr"$\theta = {float(key)}$"

        if key in errs_c:
            Ec = np.array(errs_c[key], dtype=float)
            ax_c.loglog(hs, Ec, "o-", color=col, linewidth=2, markersize=7, label=label)

        if key in errs_grad:
            Eg = np.array(errs_grad[key], dtype=float)
            ax_grad.loglog(hs, Eg, "s--", color=col, linewidth=2, markersize=7, label=label)

    # Reference lines: black, anchored at first point of key_ref
    if key_ref in errs_c:
        _add_ref_line(ax_c, hs, np.array(errs_c[key_ref], dtype=float)[0],
                      slope=int(l) + 1, x_label="h")

    if key_ref in errs_grad:
        _add_ref_line(ax_grad, hs, np.array(errs_grad[key_ref], dtype=float)[0],
                      slope=int(l), x_label="h")

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]
    method_tag = "BDF" if is_bdf else "Theta"

    for ax, all_e, norm_label, title_label in [
        (ax_c,    all_ec,    norm_label_c,    label_c),
        (ax_grad, all_egrad, norm_label_grad, label_grad),
    ]:
        ax.set_title(fr"Errors ${title_label}$  —  $\ell = {int(l)}$  —  {method_tag} saturation",
                     fontsize=16)
        ax.set_xlabel(r"$h\;[-]$",  fontsize=14)
        ax.set_ylabel(norm_label,   fontsize=14)
        ax.set_xlim(hs.min(),       hs.max())
        ax.set_ylim(min(all_e)*0.5, max(all_e)*2.0)
        ax.grid(True, which="both", linestyle=":", alpha=0.7)
        ax.legend(fontsize=12)

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            Path(__file__).parent / "Plots" / f"space_saturation_l{int(l)}_{method_tag}_{timestamp}.png",
            f"Space saturation plot (l={int(l)}, {method_tag})",
        )

    plt.show()

def plot_polynomial_saturation(l_list, errs_c, errs_grad, h: float,
                               method=TimeMethod.BDF, space_method=SpaceMethod.LDG, save=False):
    """
    Plots a polynomial saturation study: fixed mesh size h, varying time
    integration order (BDF order or Theta value), polynomial degree on x-axis.

    The reference line h^l is anchored at the first point of the highest
    BDF order (or implicit Theta), which saturates latest.

    Parameters
    ----------
    l_list       : list of PolyDegree — polynomial degrees tested (shared x-axis)
    errs_c       : dict {BdfOrder | ThetaMethod: [e0, e1, ...]}  — primal errors
    errs_grad    : dict {BdfOrder | ThetaMethod: [e0, e1, ...]}  — gradient errors
    h            : float — fixed mesh size used in all runs
    method       : TimeMethod.BDF or TimeMethod.THETA
    space_method : SpaceMethod used to pick axis / legend labels
    save         : if True, save the figure to Plots/
    """
    label_c,      label_grad      = ERROR_LABELS_PLOT[space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[space_method]

    l_ints    = [int(l) for l in l_list]
    L_num     = np.array(l_ints)
    color_map = TIME_COLORS[method]
    is_bdf    = (method == TimeMethod.BDF)
    keys      = sorted(set(errs_c) | set(errs_grad), key=lambda x: float(x))
    key_ref   = max(keys, key=lambda x: float(x))

    fig, (ax_c, ax_grad) = plt.subplots(1, 2, figsize=(15, 6))

    for key in keys:
        col   = color_map.get(key, "gray")
        label = fr"BDF{int(key)}" if is_bdf else fr"$\theta = {float(key)}$"

        if key in errs_c:
            Ec = np.array(errs_c[key], dtype=float)
            ax_c.semilogy(L_num, Ec, "o-", color=col, linewidth=2, markersize=7, label=label)

        if key in errs_grad:
            Eg = np.array(errs_grad[key], dtype=float)
            ax_grad.semilogy(L_num, Eg, "s--", color=col, linewidth=2, markersize=7, label=label)

    # Reference line h^l anchored at first point of key_ref
    L_ext = np.arange(l_ints[0], l_ints[-1] + 1)

    if key_ref in errs_c:
        e0_c = np.array(errs_c[key_ref], dtype=float)[0]
        ref_c = e0_c * (h ** (L_ext - l_ints[0]))
        ax_c.semilogy(L_ext, ref_c, "k--", linewidth=1.5, alpha=0.6, label=r"$h^{\ell}$")

    if key_ref in errs_grad:
        e0_g = np.array(errs_grad[key_ref], dtype=float)[0]
        ref_g = e0_g * (h ** (L_ext - l_ints[0]))
        ax_grad.semilogy(L_ext, ref_g, "k--", linewidth=1.5, alpha=0.6, label=r"$h^{\ell}$")

    all_ec    = [v for vals in errs_c.values()    for v in vals]
    all_egrad = [v for vals in errs_grad.values() for v in vals]
    method_tag = "BDF" if is_bdf else "Theta"

    for ax, all_e, norm_label, title_label in [
        (ax_c,    all_ec,    norm_label_c,    label_c),
        (ax_grad, all_egrad, norm_label_grad, label_grad),
    ]:
        ax.set_title(fr"Errors ${title_label}$  —  $h = {h}$  —  {method_tag} saturation",
                     fontsize=16)
        ax.set_xlabel(r"$\ell$",    fontsize=14)
        ax.set_ylabel(norm_label,   fontsize=14)
        ax.set_xlim(l_ints[0],      l_ints[-1])
        ax.set_ylim(min(all_e)*0.5, max(all_e)*2.0)
        ax.set_xticks(L_num)
        ax.grid(True, which="both", linestyle=":", alpha=0.7)
        ax.legend(fontsize=12)

    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        _save_plot(
            Path(__file__).parent / "Plots" / f"poly_saturation_{method_tag}_h{h}_{timestamp}.png",
            f"Polynomial saturation plot ({method_tag}, h={h})",
        )

    plt.show()