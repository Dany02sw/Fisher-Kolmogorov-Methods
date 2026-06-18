import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib  import Path
from typing   import List, Tuple, Optional

from fisher_kolmogorov.utilities.enum_utilities       import ConvType, StudyType, TimeMethod, PolyDegree
from fisher_kolmogorov.utilities.dictionary_utilities import ERROR_LABELS_PLOT, NORM_LABELS
from fisher_kolmogorov.plots._primitives              import save_plot
from fisher_kolmogorov.plots._axes                    import finalize_ax
from fisher_kolmogorov.plots._curves                  import (
    plot_spatial_curves,
    plot_poly_curves,
    plot_time_curves,
    plot_saturation_curves_loglog,
    plot_saturation_curves_semilogy,
)
from fisher_kolmogorov.plots.plot_utilities              import POLY_COLORS, TIME_COLORS
from fisher_kolmogorov.configs.plot_configs.subplot_spec import SubplotSpec, PlotSpec, ErrorComponent


# Layout helpers __________________________________________________________________

def _expand_axes(subplots: list, axes_flat) -> List[Tuple]:
    """Map each SubplotSpec to its one or two matplotlib Axes.

    Returns a list of (spec, ax_primary, ax_secondary_or_None) tuples.
    ax_secondary is set only for ErrorComponent.BOTH specs.
    """
    result  = []
    ax_iter = iter(axes_flat)
    for spec in subplots:
        ax_c = next(ax_iter)
        ax_g = next(ax_iter) if spec.error is ErrorComponent.BOTH else None
        result.append((spec, ax_c, ax_g))
    return result


def _physical_cols(subplots: list, spec_cols: int) -> int:
    """Compute the number of physical columns for a single layout row.

    Each BOTH spec expands to two physical columns; BASE and GRAD stay one.
    The spec_cols value is the number of SubplotSpec entries per row.
    """
    row_specs = subplots[:spec_cols]
    return sum(s.n_axes() for s in row_specs)


def _build_figure(spec: PlotSpec):
    """Create a figure and a flat list of axes matching the physical grid.

    The logical grid is (rows, cols) in SubplotSpec units. Each BOTH spec
    expands to two adjacent physical columns. All rows are assumed to have
    the same column structure as the first row.
    """
    rows, cols = spec.layout
    phys_cols  = _physical_cols(spec.subplots, cols)
    n_phys     = rows * phys_cols

    figsize = spec.figsize if spec.figsize is not None else (7.5 * phys_cols, 6 * rows)

    fig, axes = plt.subplots(rows, phys_cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    if len(axes_flat) != n_phys:
        raise ValueError(
            f"Layout mismatch: expected {n_phys} physical axes, got {len(axes_flat)}."
        )

    return fig, axes_flat


def _filter_dict(d: dict, keys: Optional[list]) -> dict:
    """Return a subset of d restricted to keys, or d itself if keys is None."""
    if keys is None:
        return d
    return {k: v for k, v in d.items() if k in keys}


# Per-study renderers _____________________________________________________________

def _render_spatial(spec: SubplotSpec, ax_c, ax_g):
    """Draw spatial convergence curves onto ax_c and/or ax_g."""
    data                          = spec.data
    label_c, label_grad           = ERROR_LABELS_PLOT[spec.space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[spec.space_method]
    hs = np.array(data.hs, dtype=float)

    # Filter to a single degree when requested, otherwise use all degrees
    if spec.poly_degree is not None:
        all_errs_c    = {spec.poly_degree: data.errs_c_space[spec.poly_degree]}
        all_errs_grad = {spec.poly_degree: data.errs_grad_space[spec.poly_degree]}
    else:
        all_errs_c    = data.errs_c_space
        all_errs_grad = data.errs_grad_space

    # Pass None for the unwanted panel; plot_spatial_curves skips None axes
    errs_c    = all_errs_c    if spec.error is not ErrorComponent.GRAD else {}
    errs_grad = all_errs_grad if spec.error is not ErrorComponent.BASE else {}
    out_ax_c  = ax_c          if spec.error is not ErrorComponent.GRAD else None
    out_ax_g  = ax_g          if spec.error is not ErrorComponent.BASE else None

    # When error=GRAD the single allocated axis is ax_c; redirect it to the grad slot
    if spec.error is ErrorComponent.GRAD:
        out_ax_g = ax_c

    plot_spatial_curves(
        out_ax_c, out_ax_g, hs, errs_c, errs_grad,
        POLY_COLORS, use_triangles=spec.use_triangles,
    )

    _finalize_spatial_axes(
        spec, out_ax_c, out_ax_g, hs,
        errs_c, errs_grad,
        label_c, label_grad, norm_label_c, norm_label_grad,
    )


def _render_polynomial(spec: SubplotSpec, ax_c, ax_g):
    """Draw polynomial convergence curves onto ax_c and/or ax_g."""
    data                          = spec.data
    label_c, label_grad           = ERROR_LABELS_PLOT[spec.space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[spec.space_method]

    h      = spec.fixed_h if spec.fixed_h is not None else data.poly_conv_h
    l_list = data.poly_conv_degrees
    l_ints = [int(l) for l in l_list]

    ec = data.errs_c_poly    if spec.error is not ErrorComponent.GRAD else []
    eg = data.errs_grad_poly if spec.error is not ErrorComponent.BASE else []

    # Polynomial plots combine both errors on a single axis by convention
    plot_poly_curves(ax_c, l_ints, ec, eg, label_c, label_grad, h)

    all_e = np.concatenate([np.array(ec, dtype=float), np.array(eg, dtype=float)])
    finalize_ax(
        ax_c, all_e,
        norm_label = f"{norm_label_c}  /  {norm_label_grad}",
        title      = fr"Errors ${label_c}$ and ${label_grad}$",
        xlabel     = r"$\ell$",
        legend_loc = "upper right",
    )


def _render_temporal(spec: SubplotSpec, ax_c, ax_g):
    """Draw temporal convergence curves onto ax_c and/or ax_g.

    Filters to spec.time_keys if provided; otherwise plots all available orders.
    """
    data                          = spec.data
    label_c, label_grad           = ERROR_LABELS_PLOT[spec.space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[spec.space_method]

    is_bdf     = (spec.time_method is TimeMethod.BDF)
    method_tag = "BDF" if is_bdf else "Theta"
    dt_arr     = np.array(data.dt, dtype=float)
    color_map  = TIME_COLORS[spec.time_method]

    raw_c    = data.errs_c_bdf    if is_bdf else data.errs_c_theta
    raw_grad = data.errs_grad_bdf if is_bdf else data.errs_grad_theta

    # Filter to requested time keys when provided
    errs_c    = _filter_dict(raw_c,    spec.time_keys) if spec.error is not ErrorComponent.GRAD else {}
    errs_grad = _filter_dict(raw_grad, spec.time_keys) if spec.error is not ErrorComponent.BASE else {}

    # When error=GRAD the single allocated axis is ax_c; redirect to grad slot
    if spec.error is ErrorComponent.GRAD:
        plot_time_curves(None, ax_c, dt_arr, {}, errs_grad,
                         spec.time_method, color_map,
                         use_triangles=spec.use_triangles)
        all_eg = [v for vals in errs_grad.values() for v in vals]
        ax_c.set_xlim(dt_arr.min(), dt_arr.max())
        finalize_ax(
            ax_c, np.array(all_eg, dtype=float),
            norm_label = norm_label_grad,
            title      = fr"Errors ${label_grad}$ — {method_tag}",
            xlabel     = r"$\tau\;[-]$",
            legend_loc = "lower right",
        )
        return

    plot_time_curves(
        ax_c, ax_g, dt_arr, errs_c, errs_grad,
        spec.time_method, color_map, use_triangles=spec.use_triangles,
    )

    all_ec = [v for vals in errs_c.values() for v in vals]
    ax_c.set_xlim(dt_arr.min(), dt_arr.max())
    finalize_ax(
        ax_c, np.array(all_ec, dtype=float),
        norm_label = norm_label_c,
        title      = fr"Errors ${label_c}$ — {method_tag}",
        xlabel     = r"$\tau\;[-]$",
        legend_loc = "lower right",
    )

    if ax_g is not None:
        all_eg = [v for vals in errs_grad.values() for v in vals]
        ax_g.set_xlim(dt_arr.min(), dt_arr.max())
        finalize_ax(
            ax_g, np.array(all_eg, dtype=float),
            norm_label = norm_label_grad,
            title      = fr"Errors ${label_grad}$ — {method_tag}",
            xlabel     = r"$\tau\;[-]$",
            legend_loc = "lower right",
        )


def _render_spatial_saturation(spec: SubplotSpec, ax_c, ax_g):
    """Draw spatial saturation curves onto ax_c and/or ax_g.

    Filters by spec.poly_degree (single degree) or spec.degrees (multi-degree),
    and by spec.time_keys within each degree block.
    """
    data                          = spec.data
    label_c, label_grad           = ERROR_LABELS_PLOT[spec.space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[spec.space_method]

    hs         = np.array(data.space_sat_hs, dtype=float)
    color_map  = TIME_COLORS[spec.time_method]
    method_tag = "BDF" if spec.time_method is TimeMethod.BDF else "Theta"

    if spec.degrees is not None:
        # Combined multi-degree saturation — caller must provide matching axes
        for idx, l in enumerate(spec.degrees):
            ec    = _filter_dict(data.errs_c_space_sat[l],    spec.time_keys)
            eg    = _filter_dict(data.errs_grad_space_sat[l], spec.time_keys)
            _ax_c = ax_c[idx] if hasattr(ax_c, "__len__") else ax_c
            _ax_g = ax_g[idx] if hasattr(ax_g, "__len__") else ax_g
            plot_saturation_curves_loglog(
                _ax_c, _ax_g, hs, ec, eg,
                time_method    = spec.time_method,
                color_map      = color_map,
                ref_slope_c    = int(l) + 1,
                ref_slope_grad = int(l),
                x_label        = "h",
            )
            _finalize_sat_ax_pair(
                _ax_c, _ax_g, ec, eg, hs,
                label_c, label_grad, norm_label_c, norm_label_grad,
                l, method_tag,
            )
        return

    l  = spec.poly_degree if spec.poly_degree is not None else PolyDegree.P2
    ec = _filter_dict(data.errs_c_space_sat[l],    spec.time_keys)
    eg = _filter_dict(data.errs_grad_space_sat[l], spec.time_keys)

    out_ax_c = ax_c if spec.error is not ErrorComponent.GRAD else None
    out_ax_g = ax_g if spec.error is not ErrorComponent.BASE else None
    if spec.error is ErrorComponent.GRAD:
        out_ax_g = ax_c

    plot_saturation_curves_loglog(
        out_ax_c, out_ax_g, hs, ec, eg,
        time_method    = spec.time_method,
        color_map      = color_map,
        ref_slope_c    = int(l) + 1,
        ref_slope_grad = int(l),
        x_label        = "h",
    )
    _finalize_sat_ax_pair(
        out_ax_c, out_ax_g, ec, eg, hs,
        label_c, label_grad, norm_label_c, norm_label_grad,
        l, method_tag,
    )


def _render_polynomial_saturation(spec: SubplotSpec, ax_c, ax_g):
    """Draw polynomial saturation curves onto ax_c and/or ax_g.

    Filters to spec.time_keys if provided; otherwise plots all available orders.
    """
    data                          = spec.data
    label_c, label_grad           = ERROR_LABELS_PLOT[spec.space_method]
    norm_label_c, norm_label_grad = NORM_LABELS[spec.space_method]

    h          = spec.fixed_h if spec.fixed_h is not None else data.poly_sat_h
    l_list     = data.poly_sat_degrees
    l_ints     = [int(l) for l in l_list]
    color_map  = TIME_COLORS[spec.time_method]
    method_tag = "BDF" if spec.time_method is TimeMethod.BDF else "Theta"

    # Filter to requested time keys when provided
    errs_c_sat    = _filter_dict(data.errs_c_poly_sat,    spec.time_keys)
    errs_grad_sat = _filter_dict(data.errs_grad_poly_sat, spec.time_keys)

    out_ax_c = ax_c if spec.error is not ErrorComponent.GRAD else None
    out_ax_g = ax_g if spec.error is not ErrorComponent.BASE else None
    if spec.error is ErrorComponent.GRAD:
        out_ax_g = ax_c

    plot_saturation_curves_semilogy(
        out_ax_c, out_ax_g, l_ints,
        errs_c_sat, errs_grad_sat,
        spec.time_method, color_map, h,
    )

    for ax, errs, norm_label, label in (
        (out_ax_c, errs_c_sat,    norm_label_c,    label_c),
        (out_ax_g, errs_grad_sat, norm_label_grad, label_grad),
    ):
        if ax is None:
            continue
        all_e = [v for vals in errs.values() for v in vals]
        ax.set_xlim(l_ints[0], l_ints[-1])
        finalize_ax(
            ax, np.array(all_e, dtype=float),
            norm_label = norm_label,
            title      = fr"Errors ${label}$  —  $h = {h}$  —  {method_tag} saturation",
            xlabel     = r"$\ell$",
            legend_loc = "best",
        )


# Finalize helpers ________________________________________________________________

def _finalize_spatial_axes(spec, ax_c, ax_g, hs,
                           errs_c, errs_grad,
                           label_c, label_grad,
                           norm_label_c, norm_label_grad):
    """Apply xlim and finalize_ax for spatial convergence panels."""
    if ax_c is not None and errs_c:
        all_ec = [v for vals in errs_c.values() for v in vals]
        ax_c.set_xlim(hs.min(), hs.max())
        finalize_ax(
            ax_c, np.array(all_ec, dtype=float),
            norm_label = norm_label_c,
            title      = fr"Errors ${label_c}$",
            xlabel     = r"$h\;[-]$",
            legend_loc = "lower right",
        )

    if ax_g is not None and errs_grad:
        all_eg = [v for vals in errs_grad.values() for v in vals]
        ax_g.set_xlim(hs.min(), hs.max())
        finalize_ax(
            ax_g, np.array(all_eg, dtype=float),
            norm_label = norm_label_grad,
            title      = fr"Errors ${label_grad}$",
            xlabel     = r"$h\;[-]$",
            legend_loc = "lower right",
        )


def _finalize_sat_ax_pair(ax_c, ax_g, ec, eg, hs,
                          label_c, label_grad,
                          norm_label_c, norm_label_grad,
                          l, method_tag):
    """Apply xlim and finalize_ax for a saturation (c, grad) axis pair."""
    if ax_c is not None:
        all_ec = [v for vals in ec.values() for v in vals]
        ax_c.set_xlim(hs.min(), hs.max())
        finalize_ax(
            ax_c, np.array(all_ec, dtype=float),
            norm_label = norm_label_c,
            title      = fr"Errors ${label_c}$  —  $\ell = {int(l)}$  —  {method_tag} saturation",
            xlabel     = r"$h\;[-]$",
            legend_loc = "best",
        )

    if ax_g is not None:
        all_eg = [v for vals in eg.values() for v in vals]
        ax_g.set_xlim(hs.min(), hs.max())
        finalize_ax(
            ax_g, np.array(all_eg, dtype=float),
            norm_label = norm_label_grad,
            title      = fr"Errors ${label_grad}$  —  $\ell = {int(l)}$  —  {method_tag} saturation",
            xlabel     = r"$h\;[-]$",
            legend_loc = "best",
        )


# Renderer dispatch tables ________________________________________________________

_CONVERGENCE_RENDERERS = {
    ConvType.SPATIAL:    _render_spatial,
    ConvType.POLYNOMIAL: _render_polynomial,
    ConvType.TEMPORAL:   _render_temporal,
}

_SATURATION_RENDERERS = {
    ConvType.SPATIAL:    _render_spatial_saturation,
    ConvType.POLYNOMIAL: _render_polynomial_saturation,
}


# Public API ______________________________________________________________________

def plot_custom(spec: PlotSpec):
    """Render an arbitrary figure from a PlotSpec.

    Each SubplotSpec in spec.subplots is rendered onto its allocated axes.
    ErrorComponent.BOTH expands to two adjacent physical axes automatically;
    the logical layout (rows, cols) is expressed in SubplotSpec units.

    Parameters
    ----------
    spec : PlotSpec
        Full figure description including subplots, layout, save flag and
        optional output path and figsize.
    """
    fig, axes_flat = _build_figure(spec)
    mapped         = _expand_axes(spec.subplots, axes_flat)

    for subplot_spec, ax_c, ax_g in mapped:
        is_saturation = subplot_spec.study_type is StudyType.SATURATION
        renderers     = _SATURATION_RENDERERS if is_saturation else _CONVERGENCE_RENDERERS

        if subplot_spec.conv_type not in renderers:
            raise ValueError(
                f"Unsupported conv_type '{subplot_spec.conv_type}' "
                f"for study_type={subplot_spec.study_type}."
            )

        renderers[subplot_spec.conv_type](subplot_spec, ax_c, ax_g)

    plt.tight_layout()

    if spec.save:
        if spec.output_path is not None:
            path = spec.output_path
        else:
            first      = spec.subplots[0]
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
            study_tag  = "sat_" if first.study_type is StudyType.SATURATION else ""
            path       = (
                Path.cwd()
                / f"{first.space_method.name}{first.time_method.name}"
                  f"_custom_{study_tag}{timestamp}.png"
            )
        save_plot(path, "Custom convergence plot")

    plt.show()