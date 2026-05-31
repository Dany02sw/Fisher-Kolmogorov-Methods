from dolfin import *

from fisher_kolmogorov.configs.test_configs.base        import TestConfig, ConvergenceParams
from fisher_kolmogorov.utilities.enum_utilities import MeshType, MeshStructure, PolyDegree, ThetaMethod, BdfOrder

# Callable objects _________________________________________________________________________________________________________________________
def _D_factory(d_ext, mesh):
    return d_ext * Identity(2)


def _c_exact_spatial(x, t):
    c_space = 0.25 * (cos(2 * pi * x[0]) * cos(2 * pi * x[1]) + 2.0)
    return c_space * (1.0 - t)


def _c_exact_temporal(x, t):
    c_space = 0.25 * (cos(2 * pi * x[0]) * cos(2 * pi * x[1]) + 2.0)
    return c_space * exp(-t)


# Test configuration for spatial-polynomial __________________________________________________________________________________________________
COSINE_SPATIAL = TestConfig(
    name       = "Cosine — spatial/polynomial convergence",
    alpha      = Constant(1.0),
    D_factory  = _D_factory,
    c_exact    = _c_exact_spatial,
    t0         = 0.0,
    d_ext      = 1.0,
    spatial = ConvergenceParams(
        N_ref          = [2, 3, 4],
        l_space        = PolyDegree.P4,
        T              = 1e-1,
        dt             = 1e-2,
        nu_or_tht      = BdfOrder.BDF1,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.STRUCTURED,
    ),
    polynomial = ConvergenceParams(
        N_fixed        = 8,
        l_list         = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3],
        T              = 2.5e-4,
        dt             = 1e-5,
        nu_or_tht      = BdfOrder.BDF4,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)

# Test configuration for temporal ____________________________________________________________________________________________________________
COSINE_TEMPORAL = TestConfig(
    name       = "Cosine — temporal convergence",
    alpha      = Constant(1.0),
    D_factory  = _D_factory,
    c_exact    = _c_exact_temporal,
    t0         = 0.0,
    d_ext      = 1e-3,
    temporal   = ConvergenceParams(
        N_fixed        = 32,
        l_space        = PolyDegree.P4,
        T              = 2.0,
        dt_list        = [0.5, 0.25, 0.125],
        nu_or_tht      = BdfOrder.BDF5,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)
