"""
Generic forward simulation runner for user-supplied meshes.
"""

from dolfin import *

from fisher_kolmogorov.utilities.profiling_utilities    import timer
from fisher_kolmogorov.utilities.print_utilities        import print_title, print_subtitle
from fisher_kolmogorov.configs.test_configs.base        import RunConfig, SimulationParams
from fisher_kolmogorov.configs.test_configs.run         import build_run_pde_data


def run_simulation(
    solver_class : type,
    run_config   : RunConfig,
    sim_params   : SimulationParams,
    output_dir   = None,
    **solver_kwargs,
) -> None:
    """
    Run a single generic forward simulation on a user-supplied mesh.

    Parameters
    ----------
    solver_class  : type
        Uninitialised solver class built by ``make_solver_class``.
    run_config    : RunConfig
        Mesh, PDE coefficients, and initial condition.
    sim_params    : SimulationParams
        Numerical parameters for the time-stepping loop.
    output_dir    : Path or None
        Directory for XDMF solution export. If None, uses the solver default.
    **solver_kwargs
        Additional keyword arguments forwarded to the solver constructor.
    """
    print_title(run_config.name)

    mesh, alpha, D, c_0 = build_run_pde_data(run_config)

    # Set quadrature degree dynamically based on the polynomial degree
    parameters["form_compiler"]["quadrature_degree"] = sim_params.l ** 2 + 4

    print_subtitle(f"{mesh.name()}  ·  {solver_class.__name__}")

    solver = solver_class(mesh, D, alpha, c_0, **solver_kwargs)

    with timer(run_config.name):
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