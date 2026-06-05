from dolfin import *

from fisher_kolmogorov.configs.test_configs.base import TestConfig, ConvergenceParams
from fisher_kolmogorov.utilities.enum_utilities  import MeshType, MeshStructure, PolyDegree, ThetaMethod, BdfOrder


# Physical data (fixed — independent of solver and study type) ________________________________________________________________________________
def _D_factory(d_ext, mesh):
    return d_ext * Identity(2)


def _c_exact_stationary(x):
    """Stationary part of the exact solution for cosine test"""
    return 0.25 * (cos(2 * pi * x[0]) * cos(2 * pi * x[1]) + 2.0)


def _c_exact_spatial(x, t):
    """Exact solution for spatial/polynomial studies (linear in time —
    exact with BDF1/IE if no linearization is used)."""
    c_space = _c_exact_stationary(x)
    return c_space * (1.0 - t)


def _c_exact_temporal(x, t):
    """Default exact solution for temporal studies."""
    c_space = _c_exact_stationary(x)
    return c_space * exp(-t)


def make_c_exact_temporal_scaled(coeff: float):
    """
    Return a scaled temporal exact solution c_space * exp(-coeff * t).

    Use via config_kwargs={"c_exact": make_c_exact_temporal_scaled(coeff)}
    when the default coeff=1.0 causes Newton convergence issues
    (known case: SPLDG with BDF2).

    Parameters
    ----------
    coeff : float — exponent coefficient
    """
    def _c_exact(x, t):
        c_space = _c_exact_stationary(x)
        return c_space * exp(-coeff * t)
    return _c_exact


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
def make_cosine_spatial(nu_or_tht=BdfOrder.BDF1, l_space=PolyDegree.P2,
                        N_ref=None, N_list=None,
                        mesh_structure=MeshStructure.UNSTRUCTURED,
                        spatial_kwargs: dict = None,
                        config_kwargs: dict = None) -> TestConfig:
    """
    Factory for the cosine spatial convergence config.

    Known overrides:
    - Theta methods: use smaller T and dt to avoid temporal pollution,
      e.g. spatial_kwargs={"T": 1e-4, "dt": 1e-5}.
    - Linearized solvers: BDF1 default is exact on the linear profile,
      so no override needed unless explicit extrapolations are used.

    Parameters
    ----------
    nu_or_tht      : BdfOrder or ThetaMethod    (default: BDF1)
    l_space        : PolyDegree                 (default: P2)
    N_ref          : list of int or None — mesh refinement exponents;
                     mesh sizes built as 2**n by the runner.
                     Ignored if N_list is provided.  (default: [2, 3, 4])
    N_list         : list of int or None — explicit mesh sizes, bypasses
                     N_ref construction.            (default: None)
    mesh_structure : MeshStructure               (default: UNSTRUCTURED)
    spatial_kwargs : overrides for any ConvergenceParams field
    config_kwargs  : overrides for any TestConfig field
    """
    base_spatial = dict(
        N_ref          = _resolve_N(N_ref, N_list, [2, 3, 4]),
        l_space        = l_space,
        T              = 1e-1,
        dt             = 1e-2,
        nu_or_tht      = nu_or_tht,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = mesh_structure,
    )
    base_spatial.update(spatial_kwargs or {})

    base_config = dict(
        name      = f"Cosine — spatial convergence ({nu_or_tht!r})",
        alpha     = Constant(1.0),
        D_factory = _D_factory,
        c_exact   = _c_exact_spatial,
        t0        = 0.0,
        d_ext     = 1.0,
        spatial   = ConvergenceParams(**base_spatial),
    )
    base_config.update(config_kwargs or {})
    return TestConfig(**base_config)


##################################
# Polynomial convergence factory #
##################################
def make_cosine_polynomial(nu_or_tht=BdfOrder.BDF1,
                           l_list=None,
                           N_fixed=None,
                           mesh_structure=MeshStructure.UNSTRUCTURED,
                           polynomial_kwargs: dict = None,
                           config_kwargs: dict = None) -> TestConfig:
    """
    Factory for the cosine polynomial convergence config.

    Known overrides:
    - Theta methods: use smaller T and dt to avoid temporal pollution when using CN,
      e.g. spatial_kwargs={"T": 2.5e-5, "dt": 1e-6}
    - Linearized solvers: BDF1 default is exact on the linear profile,
      so no override needed unless explicit extrapolations are used.

    Parameters
    ----------
    nu_or_tht         : BdfOrder or ThetaMethod  (default: BDF1)
    l_list            : list of PolyDegree       (default: P1, P2, P3)
    N_fixed           : int or None              (default: 5)
    mesh_structure    : MeshStructure             (default: UNSTRUCTURED)
    polynomial_kwargs : overrides for ConvergenceParams fields
    config_kwargs     : overrides for TestConfig fields
    """
    base_polynomial = dict(
        N_fixed        = N_fixed if N_fixed is not None else 5,
        l_list         = l_list or [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3],
        T              = 2.5e-4,
        dt             = 1e-5,
        nu_or_tht      = nu_or_tht,
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = mesh_structure,
    )
    base_polynomial.update(polynomial_kwargs or {})

    base_config = dict(
        name       = f"Cosine — polynomial convergence ({nu_or_tht!r})",
        alpha      = Constant(1.0),
        D_factory  = _D_factory,
        c_exact    = _c_exact_spatial,
        t0         = 0.0,
        d_ext      = 1.0,
        polynomial = ConvergenceParams(**base_polynomial),
    )
    base_config.update(config_kwargs or {})
    return TestConfig(**base_config)


################################
# Temporal convergence factory #
################################
def make_cosine_temporal(nu_or_tht=BdfOrder.BDF2, l_space=PolyDegree.P2,
                         N_fixed=None,
                         mesh_structure=MeshStructure.UNSTRUCTURED,
                         temporal_kwargs: dict = None,
                         config_kwargs: dict = None) -> TestConfig:
    """
    Factory for the cosine temporal convergence config.

    Known overrides:
    - SPLDG BDF2: Newton may not converge with exp(-t); use
      config_kwargs={"c_exact": make_c_exact_temporal_scaled(coeff)}
      with a smaller coeff, 
      e.g. config_kwargs={"c_exact": make_c_exact_temporal_scaled(0.75)},
      for which the nonlinear solver works.
    - l_space should be high enough to not pollute the temporal error
      for the BDF order being studied.

    Parameters
    ----------
    nu_or_tht      : BdfOrder or ThetaMethod  (default: BDF2)
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
        mesh_type      = MeshType.UNIT_SQUARE,
        mesh_structure = mesh_structure,
    )
    base_temporal.update(temporal_kwargs or {})

    base_config = dict(
        name      = f"Cosine — temporal convergence ({nu_or_tht!r})",
        alpha     = Constant(1.0),
        D_factory = _D_factory,
        c_exact   = _c_exact_temporal,
        t0        = 0.0,
        d_ext     = 1e-3,
        temporal  = ConvergenceParams(**base_temporal),
    )
    base_config.update(config_kwargs or {})
    return TestConfig(**base_config)
