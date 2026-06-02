from dolfin import *

from fisher_kolmogorov.configs.test_configs.base import TestConfig, ConvergenceParams
from fisher_kolmogorov.utilities.enum_utilities  import MeshType, MeshStructure, PolyDegree, ThetaMethod, BdfOrder


# Callable objects ____________________________________________________________________________________________________________________________
def _D_factory(d_ext, mesh):
    return d_ext * Identity(2)


def _c_exact_spatial(x, t):
    """Exact solution for spatial/polynomial convergence (linear in time — exact with BDF1/IE unless explicit
    extrapolations are used)."""
    c_space = 0.25 * (cos(2 * pi * x[0]) * cos(2 * pi * x[1]) + 2.0)
    return c_space * (1.0 - t)


def _c_exact_temporal(x, t):
    """Exact solution for temporal convergence."""
    c_space = 0.25 * (cos(2 * pi * x[0]) * cos(2 * pi * x[1]) + 2.0)
    return c_space * exp(-0.75*t)


# Spatial — BDF _______________________________________________________________________________________________________________________________
COSINE_SPATIAL_BDF = TestConfig(
    name      = "Cosine — spatial convergence (BDF)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact_spatial,
    t0        = 0.0,
    d_ext     = 1.0,
    spatial   = ConvergenceParams(
        N_ref          = [2, 3, 4],
        l_space        = PolyDegree.P2,
        T              = 1e-1,
        dt             = 1e-2,
        nu_or_tht      = BdfOrder.BDF1,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)

# Spatial — Theta (IE; exact with linear-in-time profile) _____________________________________________________________________________________
COSINE_SPATIAL_THETA = TestConfig(
    name      = "Cosine — spatial convergence (Theta)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact_spatial,
    t0        = 0.0,
    d_ext     = 1.0,
    spatial   = ConvergenceParams(
        N_ref          = [2, 3, 4],
        l_space        = PolyDegree.P4,
        T              = 1e-1,
        dt             = 1e-2,
        nu_or_tht      = ThetaMethod.IE,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)
# COSINE_SPATIAL_THETA = COSINE_SPATIAL_BDF
# COSINE_SPATIAL_THETA.spatial.nu_or_tht = ThetaMethod.IE

# Polynomial — BDF ____________________________________________________________________________________________________________________________
COSINE_POLYNOMIAL_BDF = TestConfig(
    name      = "Cosine — polynomial convergence (BDF)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact_spatial,
    t0        = 0.0,
    d_ext     = 1.0,
    polynomial = ConvergenceParams(
        N_fixed        = 5,
        l_list         = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3],
        T              = 2.5e-4,
        dt             = 1e-5,
        nu_or_tht      = BdfOrder.BDF1,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)

# Polynomial — Theta (IE; exact with linear-in-time profile) __________________________________________________________________________________
COSINE_POLYNOMIAL_THETA = TestConfig(
    name      = "Cosine — polynomial convergence (Theta)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact_spatial,
    t0        = 0.0,
    d_ext     = 1.0,
    polynomial = ConvergenceParams(
        N_fixed        = 5,
        l_list         = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3],
        T              = 2.5e-4,
        dt             = 1e-5,
        nu_or_tht      = ThetaMethod.IE,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)

# Temporal — BDF ______________________________________________________________________________________________________________________________
COSINE_TEMPORAL_BDF = TestConfig(
    name      = "Cosine — temporal convergence (BDF)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact_temporal,
    t0        = 0.0,
    d_ext     = 1e-3,
    temporal  = ConvergenceParams(
        N_fixed        = 32,
        l_space        = PolyDegree.P2,
        T              = 2.0,
        dt_list        = [0.5, 0.25, 0.125],
        nu_or_tht      = BdfOrder.BDF2,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)

# Temporal — Theta ____________________________________________________________________________________________________________________________
COSINE_TEMPORAL_THETA = TestConfig(
    name      = "Cosine — temporal convergence (Theta)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact_temporal,
    t0        = 0.0,
    d_ext     = 1e-3,
    temporal  = ConvergenceParams(
        N_fixed        = 32,
        l_space        = PolyDegree.P2,
        T              = 2.0,
        dt_list        = [0.5, 0.25, 0.125],
        nu_or_tht      = ThetaMethod.CN,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
)