from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

from fisher_kolmogorov.utilities.enum_utilities import (
    ConvType, StudyType, SpaceMethod, TimeMethod, PolyDegree, ErrorComponent
)


# SubplotSpec _____________________________________________________________________________________________________________________________________
@dataclass
class SubplotSpec:
    """Description of a single study panel (or pair of panels when error=BOTH).

    Parameters
    ----------
    study        : ConvType      — SPATIAL, POLYNOMIAL or TEMPORAL
    error        : ErrorComponent — which error panel(s) to draw
    data         : module        — a models_config module (e.g. spldg_bdf)
    space_method : SpaceMethod
    time_method  : TimeMethod
    saturation   : bool          — True for saturation studies, False for convergence
    poly_degree  : PolyDegree    — fixed degree for spatial saturation (single-degree)
    degrees      : list          — two PolyDegree entries for combined spatial saturation
    fixed_h      : float         — fixed mesh size for polynomial studies
    fixed_dt     : float         — unused directly; reserved for future temporal specs
    use_triangles: bool          — slope triangles (True) vs reference lines (False)
    """
    study        : ConvType
    error        : ErrorComponent
    data         : object
    space_method : SpaceMethod
    time_method  : TimeMethod
    saturation   : bool                   = False
    poly_degree  : Optional[PolyDegree]   = None
    degrees      : Optional[list]         = None
    fixed_h      : Optional[float]        = None
    fixed_dt     : Optional[float]        = None
    use_triangles: bool                   = True

    def n_axes(self) -> int:
        """Return the number of matplotlib axes this spec occupies."""
        return 2 if self.error is ErrorComponent.BOTH else 1


# PlotSpec ________________________________________________________________________________________________________________________________________
@dataclass
class PlotSpec:
    """Full description of a custom figure composed of arbitrary subplots.

    Parameters
    ----------
    subplots    : list of SubplotSpec — ordered list of panels
    layout      : tuple (rows, cols) — grid shape in terms of SubplotSpec entries;
                  BOTH specs expand to two physical axes automatically
    save        : bool
    output_path : Path or None — explicit save path; if None plot_custom builds
                  one from space/time/study tags of the first subplot
    figsize     : tuple or None — passed directly to plt.subplots; if None a
                  sensible default is computed from the physical axis count
    """
    subplots    : list
    layout      : tuple
    save        : bool          = False
    output_path : Optional[Path] = None
    figsize     : Optional[tuple] = None

    def n_physical_axes(self) -> int:
        """Total number of matplotlib axes after expanding BOTH specs."""
        return sum(s.n_axes() for s in self.subplots)
