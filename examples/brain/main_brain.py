"""
Forward simulation of α-synuclein spreading on a 2-D brain section.

Edit the block delimited by CONFIGURATION / END CONFIGURATION and run:

    python main_brain.py
"""

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
from fisher_kolmogorov.models.solver_ldg_theta  import SolverLdgTheta
from fisher_kolmogorov.configs.test_configs.brain       import BRAIN_SAGITTAL
from fisher_kolmogorov.configs.model_configs.params      import LdgParams
from fisher_kolmogorov.configs.test_configs.base        import SimulationParams
from fisher_kolmogorov.utilities.enum_utilities import PolyDegree, ThetaMethod

SOLVER_CLASS = SolverLdgTheta
BRAIN_CONFIG = BRAIN_SAGITTAL
MODEL_PARAMS = LdgParams(C11=1.0, C12=0.5)
SIM_PARAMS   = SimulationParams(
    l         = PolyDegree.P2,
    T         = 50.0,
    dt        = 2.5e-1,
    nu_or_tht = ThetaMethod.CN,
    tol       = 1e-6,
    max_it    = 500,
)
# ── END CONFIGURATION ─────────────────────────────────────────────────────────

from dolfin import *

from fisher_kolmogorov.meshes.mesh_import            import mesh_factory
from fisher_kolmogorov.utilities.enum_utilities      import MeshType
from fisher_kolmogorov.utilities.initial_conditions  import get_initial_condition
from fisher_kolmogorov.utilities.profiling_utilities import timer
from fisher_kolmogorov.utilities.print_utilities     import print_title, print_subtitle
from fisher_kolmogorov.configs.test_configs.base             import BrainConfig, SimulationParams


def build_brain_pde_data(config: BrainConfig):
    """
    Assemble the spatially varying PDE coefficients for a given brain section.

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
    c_0        : initial condition
    """
    mesh, subdomains = mesh_factory(mesh_type=MeshType.BRAIN_2D, brain_plane=config.plane)
    c_0              = get_initial_condition(plane=config.plane)

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


if __name__ == "__main__":
    print_title("α-synuclein spreading — brain simulation")

    mesh, _, alpha, D, c_0 = build_brain_pde_data(BRAIN_CONFIG)

    parameters["form_compiler"]["quadrature_degree"] = SIM_PARAMS.l ** 2 + 4

    print_subtitle(f"Spreading on {mesh.name()}  ·  {SOLVER_CLASS.__name__}")

    solver = SOLVER_CLASS(mesh, D, alpha, c_0, **MODEL_PARAMS.to_kwargs())
    with timer(f"Spreading on {mesh.name()}"):
        solver.Solve(
            t0=SIM_PARAMS.t0, dt=SIM_PARAMS.dt, T=SIM_PARAMS.T,
            order=SIM_PARAMS.nu_or_tht, l=SIM_PARAMS.l,
            tol=SIM_PARAMS.tol, maxIt=SIM_PARAMS.max_it,
        )