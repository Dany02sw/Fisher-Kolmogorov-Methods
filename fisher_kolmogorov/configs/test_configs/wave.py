from dolfin import *
from ufl   import tanh

from fisher_kolmogorov.configs.test_configs.base import TestConfig, ConvergenceParams
from fisher_kolmogorov.utilities.enum_utilities  import MeshType, MeshStructure, PolyDegree, ThetaMethod, BdfOrder


# Physical data (fixed — independent of solver and study type) ________________________________________________________________________________
def _D_factory(d_ext, mesh):
    return d_ext * Identity(2)


def _c_exact(x, t):
    """Travelling-wave (tanh) exact solution."""
    alpha = Constant(1.0)
    d_ext = Constant(1e-3)
    v     = Constant(5.0 * sqrt(alpha * d_ext / 6.0))
    return 0.25 * (1.0 + tanh(8.0 - sqrt(alpha / (24.0 * d_ext)) * (x[0] - v * t))) ** 2


BOTTOM_LEFT  = Point((0.0, 0.0))
TOP_RIGHT    = Point((3.0, 1.0))
_MESH_KWARGS = {"P1": BOTTOM_LEFT, "P2": TOP_RIGHT}


# Helper ______________________________________________________________________________________________________________________________________
def _resolve_N(N_ref, N_list, default):
    """
    Resolve mesh size parameter.

    N_list takes priority over N_ref. If neither is provided the default
    is used. N_ref is stored as-is in ConvergenceParams (the runner builds
    the actual mesh sizes as 2**n internally); N_list bypasses that step.
    """
    if N_list is not None:
        return N_list
    if N_ref is not None:
        return N_ref
    return default


# Factories ___________________________________________________________________________________________________________________________________
###############################
# Spatial convergence factory #
###############################
def make_wave_spatial(nu_or_tht=BdfOrder.BDF6, l_space=PolyDegree.P2,
                      N_ref=None, N_list=None,
                      mesh_structure=MeshStructure.UNSTRUCTURED,
                      spatial_kwargs: dict = None,
                      config_kwargs: dict = None) -> TestConfig:
    """
    Factory for the wave spatial convergence config.

    Physical data (alpha, D, d_ext, domain) are fixed. Numerical parameters
    have sensible defaults but should be verified for each solver.

    The default time domain has big values to reflect the saturation study of the
    reference paper.

    Known overrides:
    - Theta methods: the wave profile is not polynomial in time and CN is
      only second-order, so a much smaller T and dt are needed to avoid
      temporal pollution, e.g. spatial_kwargs={"T": 1e-4, "dt": 1e-5}.
    - BDF default is BDF6 to keep temporal error negligible vs spatial.

    Parameters
    ----------
    nu_or_tht      : BdfOrder or ThetaMethod  (default: BDF6)
    l_space        : PolyDegree               (default: P2)
    N_ref          : list of int or None — mesh refinement exponents;
                     mesh sizes built as 2**n by the runner.
                     Ignored if N_list is provided.  (default: [3, 4, 5])
    N_list         : list of int or None — explicit mesh sizes, bypasses
                     N_ref construction.            (default: None)
    mesh_structure : MeshStructure               (default: UNSTRUCTURED)
    spatial_kwargs : overrides for ConvergenceParams fields
    config_kwargs  : overrides for TestConfig fields
    """
    base_spatial = dict(
        N_ref          = _resolve_N(N_ref, N_list, [3, 4, 5]),
        l_space        = l_space,
        T              = 1e-1,
        dt             = 1e-2,
        nu_or_tht      = nu_or_tht,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = mesh_structure,
    )
    base_spatial.update(spatial_kwargs or {})

    base_config = dict(
        name        = f"Wave (tanh) — spatial convergence ({nu_or_tht!r})",
        alpha       = Constant(1.0),
        D_factory   = _D_factory,
        c_exact     = _c_exact,
        t0          = 0.0,
        d_ext       = 1e-3,
        spatial     = ConvergenceParams(**base_spatial),
        mesh_kwargs = _MESH_KWARGS,
    )
    base_config.update(config_kwargs or {})
    return TestConfig(**base_config)


##################################
# Polynomial convergence factory #
##################################
def make_wave_polynomial(nu_or_tht=BdfOrder.BDF4,
                         l_list=None,
                         N_fixed=None,
                         mesh_structure=MeshStructure.UNSTRUCTURED,
                         polynomial_kwargs: dict = None,
                         config_kwargs: dict = None) -> TestConfig:
    """
    Factory for the wave polynomial convergence config.

    The default time domain has big values to reflect the saturation study of the
    reference paper.

    Known overrides:
    - Theta methods: it may requires smaller dt; override nu_or_tht only if needed.

    Parameters
    ----------
    nu_or_tht         : BdfOrder or ThetaMethod  (default: BDF4)
    l_list            : list of PolyDegree       (default: P1, P2)
    N_fixed           : int or None              (default: 8)
    mesh_structure    : MeshStructure            (default: UNSTRUCTURED)
    polynomial_kwargs : overrides for ConvergenceParams fields
    config_kwargs     : overrides for TestConfig fields
    """
    base_polynomial = dict(
        N_fixed        = N_fixed if N_fixed is not None else 8,
        l_list         = l_list or [PolyDegree.P1, PolyDegree.P2],
        T              = 10.0,
        dt             = 2.5e-2,
        nu_or_tht      = nu_or_tht,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = mesh_structure,
    )
    base_polynomial.update(polynomial_kwargs or {})

    base_config = dict(
        name        = f"Wave (tanh) — polynomial convergence ({nu_or_tht!r})",
        alpha       = Constant(1.0),
        D_factory   = _D_factory,
        c_exact     = _c_exact,
        t0          = 0.0,
        d_ext       = 1e-3,
        polynomial  = ConvergenceParams(**base_polynomial),
        mesh_kwargs = _MESH_KWARGS,
    )
    base_config.update(config_kwargs or {})
    return TestConfig(**base_config)


##################################
# Polynomial convergence factory #
##################################
def make_wave_temporal(nu_or_tht=BdfOrder.BDF6, l_space=PolyDegree.P2,
                       N_fixed=None,
                       mesh_structure=MeshStructure.UNSTRUCTURED,
                       temporal_kwargs: dict = None,
                       config_kwargs: dict = None) -> TestConfig:
    """
    Factory for the wave temporal convergence config.

    Known overrides:
    - l_space should be high enough to not pollute the temporal error
      for the BDF order being studied; P2 default is conservative.

    Parameters
    ----------
    nu_or_tht      : BdfOrder or ThetaMethod  (default: BDF6)
    l_space        : PolyDegree               (default: P2)
    N_fixed        : int or None              (default: 32)
    mesh_structure : MeshStructure            (default: UNSTRUCTURED)
    temporal_kwargs: overrides for ConvergenceParams fields
    config_kwargs  : overrides for TestConfig fields
    """
    base_temporal = dict(
        N_fixed        = N_fixed if N_fixed is not None else 32,
        l_space        = l_space,
        T              = 2.0,
        dt_list        = [0.5, 0.25, 0.125],
        nu_or_tht      = nu_or_tht,
        mesh_type      = MeshType.RECTANGLE,
        mesh_structure = mesh_structure,
    )
    base_temporal.update(temporal_kwargs or {})

    base_config = dict(
        name        = f"Wave (tanh) — temporal convergence ({nu_or_tht!r})",
        alpha       = Constant(1.0),
        D_factory   = _D_factory,
        c_exact     = _c_exact,
        t0          = 0.0,
        d_ext       = 1e-3,
        temporal    = ConvergenceParams(**base_temporal),
        mesh_kwargs = _MESH_KWARGS,
    )
    base_config.update(config_kwargs or {})
    return TestConfig(**base_config)