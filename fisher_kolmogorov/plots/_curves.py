import numpy as np

from fisher_kolmogorov.utilities.enum_utilities import TimeMethod, ThetaMethod, BdfOrder
from fisher_kolmogorov.plots._primitives        import add_ref_line, add_slope_triangle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _time_label(key, is_bdf: bool) -> str:
    """Build the legend label for a BDF order or Theta value."""
    return fr"BDF{int(key)}" if is_bdf else fr"$\theta = {float(key)}$"


def _expected_time_order(key, is_bdf: bool) -> int:
    """Return the expected convergence order for a time integrator key."""
    if is_bdf:
        return int(key)
    return 2 if key == ThetaMethod.CN else 1


# ---------------------------------------------------------------------------
# Spatial convergence curves (varying h, one curve per polynomial degree)
# ---------------------------------------------------------------------------

def plot_spatial_curves(ax_c, ax_grad, hs, errs_c, errs_grad,
                        poly_colors: dict, use_triangles=True):
    """
    Draw spatial convergence curves for all polynomial degrees.

    Parameters
    ----------
    ax_c, ax_grad  : matplotlib Axes
    hs             : array-like — mesh sizes
    errs_c         : dict {PolyDegree: [e0, ...]} — primal errors
    errs_grad      : dict {PolyDegree: [e0, ...]} — gradient errors
    poly_colors    : dict {PolyDegree: str}
    use_triangles  : bool — if True draw slope triangles; if False draw ref lines
    """
    hs      = np.array(hs, dtype=float)
    degrees = sorted(set(errs_c) | set(errs_grad))

    for l in degrees:
        col = poly_colors.get(l, "black")

        if l in errs_c:
            Ec = np.array(errs_c[l], dtype=float)
            ax_c.loglog(hs, Ec, "o-", color=col, linewidth=2, markersize=7,
                        label=fr"$\ell = {int(l)}$")
            if use_triangles:
                add_slope_triangle(ax_c, hs, Ec, slope=int(l) + 1, color=col)
            else:
                add_ref_line(ax_c, hs, Ec[0], slope=int(l) + 1, x_label="h")

        if l in errs_grad:
            Eg = np.array(errs_grad[l], dtype=float)
            ax_grad.loglog(hs, Eg, "s--", color=col, linewidth=2, markersize=7,
                           label=fr"$\ell = {int(l)}$")
            if use_triangles:
                add_slope_triangle(ax_grad, hs, Eg, slope=int(l), color=col)
            else:
                add_ref_line(ax_grad, hs, Eg[0], slope=int(l), x_label="h")


# ---------------------------------------------------------------------------
# Polynomial convergence curves (varying l, fixed h)
# ---------------------------------------------------------------------------

def plot_poly_curves(ax, l_ints, errors_c, errors_grad,
                     label_c: str, label_grad: str, h: float):
    """
    Draw polynomial convergence curves and the h^l reference on a semilogy axis.

    Parameters
    ----------
    ax                    : matplotlib Axes
    l_ints                : list of int — polynomial degrees
    errors_c, errors_grad : array-like — one error value per degree
    label_c, label_grad   : str — legend labels (LaTeX)
    h                     : float — fixed mesh size for the reference line
    """
    L_num  = np.array(l_ints, dtype=float)
    L_ext  = np.arange(l_ints[0], l_ints[-1] + 1)
    Ec     = np.array(errors_c,    dtype=float)
    Eg     = np.array(errors_grad, dtype=float)

    ax.semilogy(L_num, Ec, "-o",  color="yellowgreen", linewidth=2,
                markersize=8, label=fr"${label_c}$")
    ax.semilogy(L_num, Eg, "--s", color="lime",        linewidth=2,
                markersize=8, label=fr"${label_grad}$")

    ref = Ec[0] * (h ** (L_ext - l_ints[0]))
    ax.semilogy(L_ext, ref, "k--", linewidth=1.5, alpha=0.6, label=r"$h^{\ell}$")

    ax.set_xlim(l_ints[0], l_ints[-1])
    ax.set_xticks(np.arange(l_ints[0], l_ints[-1] + 1))


# ---------------------------------------------------------------------------
# Temporal convergence curves (varying dt, one curve per time integrator)
# ---------------------------------------------------------------------------

def plot_time_curves(ax_c, ax_grad, dt_arr, errs_c, errs_grad,
                     time_method, color_map: dict, use_triangles=True):
    """
    Draw temporal convergence curves for all BDF orders or Theta values.

    Parameters
    ----------
    ax_c           : matplotlib Axes — primal variable panel
    ax_grad        : matplotlib Axes or None — gradient panel; skipped when None
    dt_arr         : array-like — time step sizes
    errs_c         : dict {BdfOrder | ThetaMethod: [e0, ...]}
    errs_grad      : dict {BdfOrder | ThetaMethod: [e0, ...]}
    time_method    : TimeMethod
    color_map      : dict {BdfOrder | ThetaMethod: str}
    use_triangles  : bool — slope triangles (all-in-one) vs ref lines (single)
    """
    dt_arr = np.array(dt_arr, dtype=float)
    is_bdf = (time_method == TimeMethod.BDF)
    keys   = sorted(set(errs_c) | set(errs_grad), key=lambda x: float(x))

    for key in keys:
        col   = color_map.get(key, "black")
        label = _time_label(key, is_bdf)
        order = _expected_time_order(key, is_bdf)

        if key in errs_c:
            Ec = np.array(errs_c[key], dtype=float)
            ax_c.loglog(dt_arr, Ec, "o-", color=col, linewidth=2, markersize=7,
                        label=label)
            if use_triangles:
                add_slope_triangle(ax_c, dt_arr, Ec, slope=order, color=col)
            else:
                add_ref_line(ax_c, dt_arr, Ec[0], slope=order, x_label="tau")

        if ax_grad is not None and key in errs_grad:
            Eg = np.array(errs_grad[key], dtype=float)
            ax_grad.loglog(dt_arr, Eg, "s--", color=col, linewidth=2, markersize=7,
                           label=label)
            if use_triangles:
                add_slope_triangle(ax_grad, dt_arr, Eg, slope=order, color=col)
            else:
                add_ref_line(ax_grad, dt_arr, Eg[0], slope=order, x_label="tau")


# ---------------------------------------------------------------------------
# Saturation curves (spatial or polynomial axis, one curve per time integrator)
# ---------------------------------------------------------------------------

def plot_saturation_curves_loglog(ax_c, ax_grad, xs, errs_c, errs_grad,
                                  time_method, color_map: dict,
                                  ref_slope_c: int, ref_slope_grad: int,
                                  x_label: str):
    """
    Draw saturation curves on loglog axes and add a single ref line anchored
    at the highest-order time integrator (which saturates latest).

    Used for both spatial saturation (xs = hs) and as the inner loop of
    plot_saturation_row.

    Parameters
    ----------
    ax_c, ax_grad          : matplotlib Axes
    xs                     : array-like — x-axis values (mesh sizes)
    errs_c, errs_grad      : dict {BdfOrder | ThetaMethod: [e0, ...]}
    time_method            : TimeMethod
    color_map              : dict
    ref_slope_c/grad       : int — reference line slopes
    x_label                : str — 'h' or 'tau'
    """
    xs     = np.array(xs, dtype=float)
    is_bdf = (time_method == TimeMethod.BDF)
    keys   = sorted(set(errs_c) | set(errs_grad), key=lambda x: float(x))
    key_ref = max(keys, key=lambda x: float(x))

    for key in keys:
        col   = color_map.get(key, "gray")
        label = _time_label(key, is_bdf)

        if key in errs_c:
            Ec = np.array(errs_c[key], dtype=float)
            ax_c.loglog(xs, Ec, "o-", color=col, linewidth=2, markersize=7,
                        label=label)

        if key in errs_grad:
            Eg = np.array(errs_grad[key], dtype=float)
            ax_grad.loglog(xs, Eg, "s--", color=col, linewidth=2, markersize=7,
                           label=label)

    if key_ref in errs_c:
        add_ref_line(ax_c, xs, np.array(errs_c[key_ref], dtype=float)[0],
                     slope=ref_slope_c, x_label=x_label)

    if key_ref in errs_grad:
        add_ref_line(ax_grad, xs, np.array(errs_grad[key_ref], dtype=float)[0],
                     slope=ref_slope_grad, x_label=x_label)


def plot_saturation_curves_semilogy(ax_c, ax_grad, l_ints, errs_c, errs_grad,
                                    time_method, color_map: dict, h: float):
    """
    Draw polynomial saturation curves on semilogy axes with an h^l ref line.

    Parameters
    ----------
    ax_c, ax_grad     : matplotlib Axes
    l_ints            : list of int — polynomial degrees (x-axis)
    errs_c, errs_grad : dict {BdfOrder | ThetaMethod: [e0, ...]}
    time_method       : TimeMethod
    color_map         : dict
    h                 : float — fixed mesh size for the h^l reference line
    """
    L_num   = np.array(l_ints, dtype=float)
    L_ext   = np.arange(l_ints[0], l_ints[-1] + 1)
    is_bdf  = (time_method == TimeMethod.BDF)
    keys    = sorted(set(errs_c) | set(errs_grad), key=lambda x: float(x))
    key_ref = max(keys, key=lambda x: float(x))

    for key in keys:
        col   = color_map.get(key, "gray")
        label = _time_label(key, is_bdf)

        if key in errs_c:
            ax_c.semilogy(L_num, np.array(errs_c[key], dtype=float),
                          "o-", color=col, linewidth=2, markersize=7, label=label)

        if key in errs_grad:
            ax_grad.semilogy(L_num, np.array(errs_grad[key], dtype=float),
                             "s--", color=col, linewidth=2, markersize=7, label=label)

    if key_ref in errs_c:
        e0_c  = np.array(errs_c[key_ref],    dtype=float)[0]
        ax_c.semilogy(L_ext, e0_c * (h ** (L_ext - l_ints[0])),
                      "k--", linewidth=1.5, alpha=0.6, label=r"$h^{\ell}$")

    if key_ref in errs_grad:
        e0_g  = np.array(errs_grad[key_ref], dtype=float)[0]
        ax_grad.semilogy(L_ext, e0_g * (h ** (L_ext - l_ints[0])),
                         "k--", linewidth=1.5, alpha=0.6, label=r"$h^{\ell}$")
