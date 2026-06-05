"""
Forward simulation runner of α-synuclein spreading on a 2-D brain section.
"""

from dolfin import *

from fisher_kolmogorov.meshes.mesh_import            import mesh_factory
from fisher_kolmogorov.utilities.enum_utilities      import MeshType
from fisher_kolmogorov.utilities.initial_conditions  import get_initial_condition
from fisher_kolmogorov.utilities.profiling_utilities import timer
from fisher_kolmogorov.utilities.print_utilities     import print_title, print_subtitle
from fisher_kolmogorov.configs.test_configs.base     import BrainConfig, SimulationParams
from fisher_kolmogorov.configs.model_configs.base    import ModelParams


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


def run_brain_simulation(
    solver_class : type,
    model_params : ModelParams,
    sim_config   : BrainConfig,
    sim_params   : SimulationParams,
    save_plot    : bool  = False,
    **solver_kwargs,
) -> None:
    """
    Run a single forward simulation on the brain geometry.

    Parameters
    ----------
    solver_class : type
        Uninitialised solver class (e.g. SolverLdgTheta).
    config       : BrainConfig
        Brain physical data and section type.
    sim_params   : SimulationParams
        Numerical parameters for the forward time-stepping loop.
    model_params : ModelParams
        Model-specific parameters.
    **solver_kwargs : dict
        Additional solver-specific keyword arguments passed to the constructor.
    """

    print_title("α-synuclein spreading — brain simulation")

    # Build geometry and PDE data using the parameterized config
    mesh, _, alpha, D, c_0 = build_brain_pde_data(sim_config)

    # Set quadrature degree dynamically based on the current simulation degree
    parameters["form_compiler"]["quadrature_degree"] = sim_params.l ** 2 + 4

    print_subtitle(f"Spreading on {mesh.name()}  ·  {solver_class.__name__}")

    # Initialize solver with parameters and eventual extra kwargs
    solver = solver_class(
        mesh, D, alpha, c_0,
        **model_params.to_kwargs(),
        **solver_kwargs,
    )

    # Solver loop with profiling
    with timer(f"Spreading on {mesh.name()}"):
        solver.Solve(
            t0=sim_params.t0, dt=sim_params.dt, T=sim_params.T,
            time_order=sim_params.nu_or_tht, l=sim_params.l,
            tol=sim_params.tol, maxIt=sim_params.max_it,
        )


# Main for documentation ___________________________________________________________________________________________________________________
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()