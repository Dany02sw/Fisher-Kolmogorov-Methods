import numpy as np


def finalize_ax(ax, all_e, norm_label, title, xlabel, legend_loc="lower right",
                fontsize_title=16, fontsize_labels=14, fontsize_legend=12):
    """
    Apply standard formatting to a convergence plot axis.

    Parameters
    ----------
    ax           : matplotlib Axes — already populated with curves
    all_e        : array-like — all error values shown on this axis, used to set ylim
    norm_label   : str — y-axis label (norm notation)
    title        : str — axis title
    xlabel       : str — x-axis label (LaTeX string)
    legend_loc   : str — legend location passed to ax.legend
    fontsize_*   : int — font sizes for title, axis labels, legend
    """
    all_e = np.asarray(all_e, dtype=float)
    ax.set_title(title,      fontsize=fontsize_title)
    ax.set_xlabel(xlabel,    fontsize=fontsize_labels)
    ax.set_ylabel(norm_label, fontsize=fontsize_labels)
    ax.set_ylim(all_e.min() * 0.5, all_e.max() * 2.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(fontsize=fontsize_legend, loc=legend_loc)


def finalize_ax_pair(ax_c, ax_grad,
                     all_ec, all_egrad,
                     norm_label_c, norm_label_grad,
                     title_c, title_grad,
                     xlabel,
                     legend_loc="lower right",
                     fontsize_title=16, fontsize_labels=14, fontsize_legend=12):
    """
    Apply standard formatting to a (primal, gradient) axis pair.

    Thin convenience wrapper around finalize_ax for the common two-panel layout.

    Parameters
    ----------
    ax_c, ax_grad         : matplotlib Axes
    all_ec, all_egrad     : array-like — all error values per axis
    norm_label_c/grad     : str — y-axis labels
    title_c, title_grad   : str — axis titles
    xlabel                : str — shared x-axis label
    legend_loc            : str
    fontsize_*            : int
    """
    finalize_ax(ax_c,    all_ec,    norm_label_c,    title_c,    xlabel,
                legend_loc, fontsize_title, fontsize_labels, fontsize_legend)
    finalize_ax(ax_grad, all_egrad, norm_label_grad, title_grad, xlabel,
                legend_loc, fontsize_title, fontsize_labels, fontsize_legend)
