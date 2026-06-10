from __future__ import annotations

from typing  import Optional
from pathlib import Path  # to avoid pylance errors

from dolfin import *

from fisher_kolmogorov.meshes.mesh_import            import mesh_factory
from fisher_kolmogorov.utilities.profiling_utilities import timer
from fisher_kolmogorov.utilities.print_utilities     import print_polynomial_rates
from fisher_kolmogorov.plots.plot_utilities          import plot_polynomial_convergence
from fisher_kolmogorov.configs.test_configs.base     import TestConfig


def run_polynomial_convergence(
    solver_class : type,
    config       : TestConfig,
    tol          : float = 1e-11,
    max_it       : int   = 200,
    save_plot    : bool  = False,
    output_dir   : Optional[Path] = None,
    **solver_kwargs,
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
    tol          : float
        Nonlinear-solver tolerance.
    max_it       : int
        Maximum nonlinear-solver iterations.
    save_plot    : bool
        Whether to save the convergence plot to disk.
    output_dir   : Path or None
        If provided, solver solutions are written to XDMF files in this
        directory at every time step. Used for model comparison studies.
    **solver_kwargs : dict
        Additional solver-specific keyword arguments (e.g., ``Linearize``) 
        passed directly to the ``solver_class`` constructor.    
    """
    p = config.polynomial

    d_ext = Constant(config.d_ext)
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
                **solver_kwargs,
            )
            E_c, E_grad, h = solver.ConvergenceTest(
                t0=config.t0, dt=p.dt, T=p.T,
                time_order=p.nu_or_tht, l=l,
                tol=tol, maxIt=max_it,
                output_dir=output_dir,
            )
            errors_base.append(E_c)
            errors_grad.append(E_grad)

    print_polynomial_rates(errors_base, errors_grad, p.l_list, method=solver.SM)
    plot_polynomial_convergence(
        errors_base, errors_grad, p.l_list, h, space_method=solver.SM, time_method=solver.TM, save=save_plot
    )
