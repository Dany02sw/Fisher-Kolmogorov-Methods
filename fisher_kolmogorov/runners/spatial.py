from dolfin import *

from fisher_kolmogorov.meshes.mesh_import            import mesh_factory
from fisher_kolmogorov.utilities.profiling_utilities import timer
from fisher_kolmogorov.utilities.print_utilities     import print_space_rates
from fisher_kolmogorov.plots.plot_utilities          import plot_spatial_convergence
from fisher_kolmogorov.configs.test_configs.base     import TestConfig
from fisher_kolmogorov.configs.model_configs.base    import ModelParams


def run_spatial_convergence(
    solver_class : type,
    config       : TestConfig,
    model_params : ModelParams,
    tol          : float = 1e-11,
    max_it       : int   = 200,
    save_plot    : bool  = False,
    **solver_kwargs,
) -> None:
    """
    Run a spatial (h-refinement) convergence study.

    For each mesh size N in ``config.spatial.N_ref`` a new solver instance
    is created, ``ConvergenceTest`` is called, and errors are collected.
    Convergence rates are printed and optionally plotted.

    Parameters
    ----------
    solver_class : type
        Uninitialised solver class (e.g. SolverDgBDF).
    config       : TestConfig
        Test configuration carrying physical data and study hyperparameters.
    model_params : ModelParams
        Model-specific parameters (penalty, stabilisation, …).
    tol          : float
        Nonlinear-solver absolute/relative tolerance.
    max_it       : int
        Maximum nonlinear-solver iterations.
    save_plot    : bool
        Whether to save the convergence plot to disk.
    **solver_kwargs : dict
        Additional solver-specific keyword arguments (e.g., ``Linearize``) 
        passed directly to the ``solver_class`` constructor.  
    """
    p = config.spatial

    parameters["form_compiler"]["quadrature_degree"] = p.l_space ** 2 + 4

    d_ext = Constant(config.d_ext)
    D     = config.D_factory(d_ext, mesh=None)
    alpha = config.alpha

    N_list        = [2 ** n for n in p.N_ref]
    errors_base   = []
    errors_grad   = []
    hs            = []
    N_el_list     = []

    with timer(f"Spatial convergence  l={p.l_space}"):
        for N in N_list:
            print(f"\n --- N = {N} ---")
            mesh, _ = mesh_factory(
                mesh_type=p.mesh_type,
                N=N,
                structure=p.mesh_structure,
                **config.mesh_kwargs,
            )
            N_el_list.append(mesh.num_cells())

            solver = solver_class(
                mesh=mesh, D=D, alpha=alpha, c_0=config.c_exact,
                **model_params.to_kwargs(),
                **solver_kwargs,
            )
            E_c, E_grad, h = solver.ConvergenceTest(
                t0=config.t0, dt=p.dt, T=p.T,
                time_order=p.nu_or_tht, l=p.l_space,
                tol=tol, maxIt=max_it,
            )
            errors_base.append(E_c)
            errors_grad.append(E_grad)
            hs.append(h)

    print_space_rates(errors_base, errors_grad, hs, N_el_list, p.l_space, method=solver.SM)
    plot_spatial_convergence(
        hs, errors_base, errors_grad, p.l_space, space_method=solver.SM, time_method=solver.TM, save=save_plot
    )
