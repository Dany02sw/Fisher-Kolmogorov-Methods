from dolfin import *
from ufl   import tanh

from fisher_kolmogorov.configs.test_configs.base import TestConfig, ConvergenceParams
from fisher_kolmogorov.utilities.enum_utilities  import MeshType, MeshStructure, PolyDegree, ThetaMethod, BdfOrder


# Callable objects ____________________________________________________________________________________________________________________________
def _D_factory(d_ext, mesh):
    return d_ext * Identity(2)


def _c_exact(x, t):
    alpha = Constant(1.0)
    d_ext = Constant(1e-3)
    v     = Constant(5.0 * sqrt(alpha * d_ext / 6.0))
    return 0.25 * (1.0 + tanh(8.0 - sqrt(alpha / (24.0 * d_ext)) * (x[0] - v * t))) ** 2


BOTTOM_LEFT = Point((0.0, 0.0))
TOP_RIGHT   = Point((3.0, 1.0))

# Spatial — BDF _______________________________________________________________________________________________________________________________
WAVE_SPATIAL_BDF = TestConfig(
    name      = "Wave (tanh) — spatial convergence (BDF)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact,
    t0        = 0.0,
    d_ext     = 1e-3,
    spatial   = ConvergenceParams(
        N_ref          = [3, 4, 5],
        l_space        = PolyDegree.P2,
        T              = 1e-1,
        dt             = 1e-2,
        nu_or_tht      = BdfOrder.BDF6,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
    mesh_kwargs = {"P1": BOTTOM_LEFT, "P2": TOP_RIGHT},
)

# Spatial — Theta _____________________________________________________________________________________________________________________________
WAVE_SPATIAL_THETA = TestConfig(
    name      = "Wave (tanh) — spatial convergence (Theta)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact,
    t0        = 0.0,
    d_ext     = 1e-3,
    spatial   = ConvergenceParams(
        N_ref          = [3, 4, 5],
        l_space        = PolyDegree.P2,
        T              = 1e-4,
        dt             = 1e-5,
        nu_or_tht      = ThetaMethod.CN,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
    mesh_kwargs = {"P1": BOTTOM_LEFT, "P2": TOP_RIGHT},
)

# Polynomial — BDF ____________________________________________________________________________________________________________________________
WAVE_POLYNOMIAL_BDF = TestConfig(
    name      = "Wave (tanh) — polynomial convergence (BDF)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact,
    t0        = 0.0,
    d_ext     = 1e-3,
    polynomial = ConvergenceParams(
        N_fixed        = 8,
        l_list         = [PolyDegree.P1, PolyDegree.P2],
        T              = 10.0,
        dt             = 2.5e-2,
        nu_or_tht      = BdfOrder.BDF4,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
    mesh_kwargs = {"P1": BOTTOM_LEFT, "P2": TOP_RIGHT},
)

# Polynomial — Theta __________________________________________________________________________________________________________________________
WAVE_POLYNOMIAL_THETA = TestConfig(
    name      = "Wave (tanh) — polynomial convergence (Theta)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact,
    t0        = 0.0,
    d_ext     = 1e-3,
    polynomial = ConvergenceParams(
        N_fixed        = 8,
        l_list         = [PolyDegree.P1, PolyDegree.P2],
        T              = 10.0,
        dt             = 2.5e-2,
        nu_or_tht      = ThetaMethod.IE,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
    mesh_kwargs = {"P1": BOTTOM_LEFT, "P2": TOP_RIGHT},
)

# Temporal — BDF ______________________________________________________________________________________________________________________________
WAVE_TEMPORAL_BDF = TestConfig(
    name      = "Wave (tanh) — temporal convergence (BDF)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact,
    t0        = 0.0,
    d_ext     = 1e-3,
    temporal  = ConvergenceParams(
        N_fixed        = 32,
        l_space        = PolyDegree.P3,
        T              = 2.0,
        dt_list        = [0.5, 0.25, 0.125],
        nu_or_tht      = BdfOrder.BDF6,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
    mesh_kwargs = {"P1": BOTTOM_LEFT, "P2": TOP_RIGHT},
)

# Temporal — Theta ____________________________________________________________________________________________________________________________
WAVE_TEMPORAL_THETA = TestConfig(
    name      = "Wave (tanh) — temporal convergence (Theta)",
    alpha     = Constant(1.0),
    D_factory = _D_factory,
    c_exact   = _c_exact,
    t0        = 0.0,
    d_ext     = 1e-3,
    temporal  = ConvergenceParams(
        N_fixed        = 32,
        l_space        = PolyDegree.P3,
        T              = 2.0,
        dt_list        = [0.5, 0.25, 0.125],
        nu_or_tht      = ThetaMethod.IE,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = MeshStructure.UNSTRUCTURED,
    ),
    mesh_kwargs = {"P1": BOTTOM_LEFT, "P2": TOP_RIGHT},
)