from dolfin import *

from fisher_kolmogorov.meshes.mesh_import            import mesh_factory
from fisher_kolmogorov.utilities.profiling_utilities import timer
from fisher_kolmogorov.utilities.print_utilities     import print_polynomial_rates
from fisher_kolmogorov.plots.plot_utilities          import plot_polynomial_convergence
from fisher_kolmogorov.configs.test_configs.base             import TestConfig
from fisher_kolmogorov.configs.model_configs.base             import ModelParams


def run_polynomial_convergence(
    solver_class : type,
    config       : TestConfig,
    model_params : ModelParams,
    tol          : float = 1e-11,
    max_it       : int   = 200,
    save_plot    : bool  = False,
) -> None:
    """
    Run a polynomial-degree (p-refinement) convergence study.

    A single mesh is built with ``config.polynomial.N_fixed`` subdivisions
    and the solver is run for each degree in ``config.polynomial.l_list``.

    Parameters
    ----------
    solver_class : type
        Uninitialised solver class (e.g. SolverLdgBDF).
    config       : TestConfig
        Test configuration carrying physical data and study hyperparameters.
    model_params : ModelParams
        Model-specific parameters.
    tol          : float
        Nonlinear-solver tolerance.
    max_it       : int
        Maximum nonlinear-solver iterations.
    save_plot    : bool
        Whether to save the convergence plot to disk.
    """
    p = config.polynomial

    d_ext = Constant(config.d_ext_conv)
    D     = config.D_factory(d_ext, mesh=None)
    alpha = config.alpha

    mesh, _ = mesh_factory(
        mesh_type=p.mesh_type,
        N=p.N_fixed,
        structure=p.mesh_structure,
        **config.mesh_kwargs,
    )

    errors_base = []
    errors_grad = []
    h         = None

    with timer("Polynomial convergence"):
        for l in p.l_list:
            print(f"\n --- l = {l} ---")
            parameters["form_compiler"]["quadrature_degree"] = l ** 2 + 4

            solver = solver_class(
                mesh=mesh, D=D, alpha=alpha, c_0=config.c_exact,
                **model_params.to_kwargs(),
            )
            E_L2, E_DG, h = solver.ConvergenceTest(
                t0=config.t0, dt=p.dt, T=p.T,
                nu=p.nu_or_tht, l=l,
                tol=tol, maxIt=max_it,
            )
            errors_base.append(E_L2)
            errors_grad.append(E_DG)

    print_polynomial_rates(errors_base, errors_grad, p.l_list, method=solver.SM)
    plot_polynomial_convergence(errors_base, errors_grad, p.l_list, h, method=solver.SM, save=save_plot)
