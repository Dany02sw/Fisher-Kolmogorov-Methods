from dolfin import *

from fisher_kolmogorov.meshes.mesh_import            import mesh_factory
from fisher_kolmogorov.utilities.profiling_utilities import timer
from fisher_kolmogorov.utilities.print_utilities     import print_time_rates
from fisher_kolmogorov.utilities.math_utilities      import get_decimals
from fisher_kolmogorov.plots.plot_utilities          import plot_time_convergence
from fisher_kolmogorov.utilities.enum_utilities      import TimeMethod
from fisher_kolmogorov.configs.test_configs.base             import TestConfig
from fisher_kolmogorov.configs.model_configs.base             import ModelParams


def run_temporal_convergence(
    solver_class : type,
    config       : TestConfig,
    model_params : ModelParams,
    tol          : float = 1e-11,
    max_it       : int   = 200,
    save_plot    : bool  = False,
) -> None:
    """
    Run a temporal (dt-refinement) convergence study.

    The time integrator is detected from ``solver.TM`` after the first solve,
    and the correct keyword argument (``nu`` for BDF, ``tht`` for theta-method)
    is forwarded automatically to ``ConvergenceTest``.

    For BDF methods, the initial window is shifted by ``(nu - 1)*dt_max`` so
    that the backward-difference stencil is fully initialised before the
    measurement window begins.

    Parameters
    ----------
    solver_class : type
        Uninitialised solver class (e.g. SolverDgBDF or SolverDgTheta).
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
    p = config.temporal

    parameters["form_compiler"]["quadrature_degree"] = p.l_space ** 2 + 4

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
    solver      = None

    # Detect whether nu_or_tht encodes BDF order or theta value
    nu_or_tht = p.nu_or_tht
    is_bdf    = isinstance(nu_or_tht, int)
    dt_max    = p.dt_list[0]

    label = f"ν={nu_or_tht}" if is_bdf else f"θ={nu_or_tht}"
    with timer(f"Temporal convergence  {label}"):
        for dt in p.dt_list:
            decimals = get_decimals(dt)
            print(f"\n --- τ = {dt:.{decimals}f} ---")

            solver = solver_class(
                mesh=mesh, D=D, alpha=alpha, c_0=config.c_exact,
                **model_params.to_kwargs(),
            )

            if is_bdf:
                # Shift the window so the BDF stencil is fully warm-started
                t0_eff = config.t0 + (nu_or_tht - 1) * dt_max
                T_eff  = p.T       + (nu_or_tht - 1) * dt_max
                E_L2, E_DG, _ = solver.ConvergenceTest(
                    t0=t0_eff, dt=dt, T=T_eff,
                    nu=nu_or_tht, l=p.l_space,
                    tol=tol, maxIt=max_it,
                )
            else:
                E_L2, E_DG, _ = solver.ConvergenceTest(
                    t0=config.t0, dt=dt, T=p.T,
                    tht=nu_or_tht, l=p.l_space,
                    tol=tol, maxIt=max_it,
                )

            errors_base.append(E_L2)
            errors_grad.append(E_DG)

    print_time_rates(
        errors_base, errors_grad, p.dt_list, nu_or_tht,
        time_method=solver.TM, space_method=solver.SM,
    )
    plot_time_convergence(
        p.dt_list, errors_base, errors_grad, nu_or_tht,
        method=solver.TM, space_method=solver.SM, save=save_plot,
    )
