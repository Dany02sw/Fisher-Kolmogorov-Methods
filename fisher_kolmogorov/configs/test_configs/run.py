from __future__ import annotations

from pathlib import Path
from typing  import Callable, Union

from dolfin import Constant, Identity

from fisher_kolmogorov.configs.test_configs.base import RunConfig
from fisher_kolmogorov.utilities.enum_utilities  import MeshType


# PDE data assembly ________________________________________________________________________________________________________________________
def build_run_pde_data(config: RunConfig):
    """
    Assemble mesh and PDE coefficients from a RunConfig.

    In simple mode (``user_mesh_path`` provided) the mesh is loaded via
    ``mesh_factory``. In advanced mode (``mesh`` provided directly) the
    mesh is used as-is; ``alpha`` and ``D`` are expected to already be
    valid UFL expressions over that mesh.

    Parameters
    ----------
    config : RunConfig

    Returns
    -------
    mesh  : dolfin Mesh
    alpha : UFL expression
    D     : UFL expression
    c_0   : callable (x, t) -> UFL expression
    """
    if config.mesh is not None:
        # Advanced mode: mesh and coefficients already built by the user
        return config.mesh, config.alpha, config.D, config.c_0

    # Simple mode: load mesh from .msh via factory
    from fisher_kolmogorov.meshes.mesh_import import mesh_factory    # local import to avoid circular deps
    mesh, _ = mesh_factory(
        mesh_type      = MeshType.USER,
        user_mesh_path = config.user_mesh_path,
    )
    return mesh, config.alpha, config.D, config.c_0


# Run configuration factory ________________________________________________________________________________________________________________
def make_run_config(
    user_mesh_path : Union[Path, str],
    c_0            : Callable,
    alpha          : float  = 1.0,
    D              : float  = 1e-2,
    name           : str    = "User simulation",
    config_kwargs  : dict   = None,
) -> RunConfig:
    """
    Factory for simple forward simulation configs with scalar PDE coefficients.

    For spatially varying coefficients (e.g. subdomain-dependent alpha or D),
    build ``RunConfig`` directly in your ``fk_config.py`` and pass a pre-built
    ``mesh`` instead of ``user_mesh_path``.

    Parameters
    ----------
    user_mesh_path : Path or str
        Path to the input .msh file. Converted to XDMF automatically.
    c_0            : callable (x, t) -> UFL expression
        Initial condition.
    alpha          : float
        Scalar reaction coefficient (default: 1.0).
    D              : float
        Scalar isotropic diffusion coefficient (default: 1e-2).
    name           : str
        Label used in print headers.
    config_kwargs  : dict, optional
        Overrides for any ``RunConfig`` field.

    Returns
    -------
    RunConfig
    """
    base = dict(
        mesh_type      = MeshType.USER,
        user_mesh_path = Path(user_mesh_path),
        alpha          = Constant(alpha),
        D              = Constant(D) * Identity(2),
        c_0            = c_0,
        name           = name,
    )
    base.update(config_kwargs or {})
    return RunConfig(**base)