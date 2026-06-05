import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def add_ref_line(ax, xs, e0, slope, x_label):
    """
    Draw a black dashed reference line anchored at the first data point.

    Follows  e(x) = e0 * (xs / xs[0])^slope  and labels the line with
    the appropriate LaTeX symbol depending on x_label.

    Parameters
    ----------
    ax      : matplotlib Axes
    xs      : array-like — x-axis values (mesh sizes or time steps)
    e0      : float — error value at xs[0], used as anchor
    slope   : int or float — exponent for the reference line
    x_label : str, 'h' or 'tau' — selects the LaTeX label symbol
    """
    xs  = np.asarray(xs, dtype=float)
    sym = r"\tau" if x_label == "tau" else "h"
    ref = e0 * (xs / xs[0]) ** slope
    ax.loglog(xs, ref, "k--", linewidth=1.5, alpha=0.6,
              label=fr"$O({sym}^{{{slope}}})$")


def add_slope_triangle(ax, xs, errs, slope, color, tri_frac=0.18):
    """
    Draw a small slope triangle near the last two points of a loglog curve.

    Triangle size scales with the log10 range of xs for visual consistency
    across different x-axis spans.

    Parameters
    ----------
    ax       : matplotlib Axes
    xs       : array-like — x-axis values
    errs     : array-like — error values
    slope    : int — exponent to annotate
    color    : str — triangle and label color
    tri_frac : float — triangle leg size as a fraction of the log10 x-range
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


def save_plot(fig_path: Path, label: str):
    """Save the current figure and print a confirmation message."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"{label} saved into {fig_path} folder!")
