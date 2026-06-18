from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import List, Optional

from fisher_kolmogorov.utilities.enum_utilities import (
    ConvType, StudyType, SpaceMethod, TimeMethod, PolyDegree, ErrorComponent,
)


# SubplotSpec _____________________________________________________________________

@dataclass
class SubplotSpec:
    """Description of a single study panel (or pair of panels when error=BOTH).

    Parameters
    ----------
    conv_type    : ConvType       — SPATIAL, POLYNOMIAL or TEMPORAL
    error        : ErrorComponent — which error panel(s) to draw
    data         : module         — a models_config module (e.g. spldg_bdf)
    space_method : SpaceMethod
    time_method  : TimeMethod
    study_type   : StudyType      — CONVERGENCE or SATURATION
    poly_degree  : PolyDegree     — single degree filter for spatial convergence
                                    and spatial saturation (single-degree path)
    degrees      : list           — multi-degree list for combined spatial saturation
    time_keys    : list           — subset of BdfOrder or ThetaMethod keys to plot;
                                    applies to temporal convergence and both saturation
                                    renderers. If None, all available keys are plotted.
    fixed_h      : float          — fixed mesh size for polynomial studies
    fixed_dt     : float          — reserved for future temporal specs
    use_triangles: bool           — slope triangles (True) vs reference lines (False)
    """

    conv_type    : ConvType
    error        : ErrorComponent
    data         : object
    space_method : SpaceMethod
    time_method  : TimeMethod
    study_type   : StudyType            = StudyType.CONVERGENCE
    poly_degree  : Optional[PolyDegree] = None
    degrees      : Optional[List]       = None
    time_keys    : Optional[List]       = None
    fixed_h      : Optional[float]      = None
    fixed_dt     : Optional[float]      = None
    use_triangles: bool                 = True

    def n_axes(self) -> int:
        """Return the number of matplotlib axes this spec occupies."""
        return 2 if self.error is ErrorComponent.BOTH else 1


# PlotSpec ________________________________________________________________________

@dataclass
class PlotSpec:
    """Full description of a custom figure composed of arbitrary subplots.

    Parameters
    ----------
    subplots    : list of SubplotSpec — ordered list of panels
    layout      : tuple (rows, cols)  — grid shape in SubplotSpec units;
                  BOTH specs expand to two physical axes automatically
    save        : bool
    output_path : Path or None        — explicit save path; if None, plot_custom
                  builds one from space/time/study tags of the first subplot
    figsize     : tuple or None       — passed to plt.subplots; if None a default
                  is computed from the physical axis count
    """

    subplots    : list
    layout      : tuple
    save        : bool            = False
    output_path : Optional[Path]  = None
    figsize     : Optional[tuple] = None

    def n_physical_axes(self) -> int:
        """Total number of matplotlib axes after expanding BOTH specs."""
        return sum(s.n_axes() for s in self.subplots)