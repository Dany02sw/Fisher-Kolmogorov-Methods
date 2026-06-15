from fisher_kolmogorov.configs.plot_configs.subplot_spec import PlotSpec, SubplotSpec

from fisher_kolmogorov.utilities.enum_utilities import (
    ConvType, StudyType, SpaceMethod, TimeMethod, ErrorComponent, PolyDegree
)

from fisher_kolmogorov.plots.plot_custom import plot_custom

from examples.plots.models_config import spldg_bdf

plot1 = SubplotSpec(
    conv_type    = ConvType.SPATIAL,
    error        = ErrorComponent.GRAD,
    data         = spldg_bdf,
    space_method = SpaceMethod.SPLDG,
    time_method  = TimeMethod.BDF,
    study_type   = StudyType.CONVERGENCE,
    poly_degree  = PolyDegree.P4
)
plot2 = SubplotSpec(
    conv_type    = ConvType.POLYNOMIAL,
    error        = ErrorComponent.BOTH,
    data         = spldg_bdf,
    space_method = SpaceMethod.SPLDG,
    time_method  = TimeMethod.BDF,
    study_type   = StudyType.SATURATION
)
plot_specs = PlotSpec(
    subplots=[plot1, plot2],
    layout=(1,2),
)
plot_custom(plot_specs)