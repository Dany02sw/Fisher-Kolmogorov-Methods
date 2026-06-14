"""
Forward simulation runner of α-synuclein spreading on a 2-D brain section.
"""

from dolfin import *

from fisher_kolmogorov.utilities.profiling_utilities import timer
from fisher_kolmogorov.utilities.print_utilities     import print_title, print_subtitle
from fisher_kolmogorov.configs.test_configs.base     import BrainConfig, SimulationParams
from fisher_kolmogorov.configs.test_configs.brain    import build_brain_pde_data


def run_brain_simulation(
    solver_class : type,
    sim_config   : BrainConfig,
    sim_params   : SimulationParams,
    output_dir   = None,
    **solver_kwargs,
) -> None:
    """
    Run a single forward simulation on the brain geometry.

    Parameters
    ----------
    solver_class : type
        Uninitialised solver class (e.g. SolverLdgTheta).
    sim_config   : BrainConfig
        Brain physical data and section type.
    sim_params   : SimulationParams
        Numerical parameters for the forward time-stepping loop.
    output_dir   : Path or None
        Directory for XDMF solution export. If None, uses the solver default.
    **solver_kwargs
        Additional keyword arguments passed to the solver constructor.
    """
    print_title("α-synuclein spreading — brain simulation")

    # Build geometry and PDE data using the parameterized config
    mesh, _, alpha, D, c_0 = build_brain_pde_data(sim_config)

    # Set quadrature degree dynamically based on the current simulation degree
    parameters["form_compiler"]["quadrature_degree"] = sim_params.l ** 2 + 4

    print_subtitle(f"Spreading on {mesh.name()}  ·  {solver_class.__name__}")

    # Initialize solver with parameters and eventual extra kwargs
    solver = solver_class(mesh, D, alpha, c_0, **solver_kwargs)

    # Solver loop with profiling
    with timer(f"Spreading on {mesh.name()}"):
        solve_kwargs = dict(
            t0         = sim_params.t0,
            dt         = sim_params.dt,
            T          = sim_params.T,
            time_order = sim_params.nu_or_tht,
            l          = sim_params.l,
            tol        = sim_params.tol,
            maxIt      = sim_params.max_it,
        )
        if output_dir is not None:
            solve_kwargs["output_dir"] = output_dir

        solver.Solve(**solve_kwargs)
