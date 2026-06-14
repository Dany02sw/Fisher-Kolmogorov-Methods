from dolfin import *
from ufl    import tanh

from fisher_kolmogorov.configs.test_configs.base import BrainConfig
from fisher_kolmogorov.utilities.enum_utilities  import BrainSection


# Initial conditions ____________________________________________________________________________________________________________________________________________
def _get_initial_condition(plane: BrainSection):
    """
    Return the initial condition callable for a given brain section.

    The initial condition models a localised seed of α-synuclein as a
    smooth bump centred on a known injection site for each anatomical plane.

    Parameters
    ----------
    plane : BrainSection

    Returns
    -------
    c_0 : callable (x, t) -> UFL expression
    """
    R0   = Constant(2.0)
    beta = Constant(1.0)

    if plane == BrainSection.SAGITTAL:
        xc, yc = -16.0, 3.5
        radius = lambda x: sqrt(pow((x[0] - xc), 2) + pow(1*(x[1] - yc), 2))
    elif plane == BrainSection.CORONAL:
        xc, yc = 8.0, -0.02
        radius = lambda x: sqrt(pow((x[0] - xc), 2) + pow(5*(x[1] - yc), 2))
    elif plane == BrainSection.HORIZONTAL:
        xc, yc = 0.0, 0.0
        radius = lambda x: sqrt(pow((x[0] - xc), 2) + pow((x[1] - yc), 2))
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return lambda x, t: 0.25*(1.0 + tanh(4.0 - beta*(radius(x) - R0)))**2


# PDE data assembly ______________________________________________________________________________________________________________________________________________
def build_brain_pde_data(config: BrainConfig):
    """
    Assemble the spatially varying PDE coefficients for a given brain section.

    The diffusion tensor and reaction coefficient are piecewise constant over
    grey and white matter subdomains, reconstructed from the mesh subdomain
    markers via a DG0 function.

    Parameters
    ----------
    config : BrainConfig
        Physical parameters and brain section for the simulation.

    Returns
    -------
    mesh       : dolfin Mesh
    subdomains : dolfin MeshFunction
    alpha      : UFL expression for the reaction coefficient
    D          : UFL expression for the diffusion tensor
    c_0        : callable (x, t) -> UFL expression
    """
    from fisher_kolmogorov.meshes.mesh_import import mesh_factory      # local import to avoid circular deps
    from fisher_kolmogorov.utilities.enum_utilities import MeshType    # local import to avoid circular deps

    mesh, subdomains = mesh_factory(mesh_type=MeshType.BRAIN_2D, brain_plane=config.plane)
    c_0              = _get_initial_condition(plane=config.plane)

    DG0                         = FunctionSpace(mesh, "DG", 0)
    subdomains_tags             = Function(DG0)
    subdomains_tags.vector()[:] = subdomains.array()

    alpha_grey  = Constant(config.alpha_grey)
    alpha_white = Constant(config.alpha_white)
    alpha       = conditional(le(subdomains_tags, 1.5), alpha_grey, alpha_white)

    a       = as_tensor(config.a)
    d_ext   = Constant(config.d_ext)
    d_axn   = Constant(config.d_axn)
    D_grey  = d_ext * Identity(2)
    D_white = d_ext * Identity(2) + d_axn * outer(a, a)
    D       = conditional(le(subdomains_tags, 1.5), D_grey, D_white)

    return mesh, subdomains, alpha, D, c_0


# Brain configuration factory ____________________________________________________________________________________________________________________________________
def make_brain_config(
    section       : BrainSection = BrainSection.SAGITTAL,
    config_kwargs : dict         = None,
) -> BrainConfig:
    """
    Factory for brain simulation configs.

    Override any field via config_kwargs when
    running with different physical assumptions.

    Parameters
    ----------
    section       : BrainSection  (default: SAGITTAL)
    config_kwargs : overrides for any BrainConfig field
                    (alpha_grey, alpha_white, d_ext, d_axn, a)
    """
    base = dict(
        plane       = section,
        alpha_grey  = 0.5,
        alpha_white = 1.0,
        d_ext       = 8e-3,
        d_axn       = 8e-2,
        a           = (1.0, 1.0),
    )
    base.update(config_kwargs or {})
    return BrainConfig(**base)
