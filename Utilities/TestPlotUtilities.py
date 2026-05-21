from Utilities.PlotUtilities import *

import numpy as np

if __name__ == "__main__":

    rng = np.random.default_rng(0)

    def _make_errors(xs, order, C=1.0):
        """Errors decaying like C * xs^order with mild noise."""
        return C * xs**order * (1 + 0.04 * rng.standard_normal(len(xs)))

    # ── Spatial convergence test data ────────────────────────────────────────
    # Four refinement levels, h halved each time
    hs_test  = np.array([1/4, 1/8, 1/16, 1/32])
    l_list   = [1, 2, 3, 4, 5, 6, 7, 8]

    # errs_c[l]    decays like h^(l+1)  — optimal primal rate
    # errs_grad[l] decays like h^l      — optimal gradient rate
    errs_c_space = {
        l: _make_errors(hs_test, order=l + 1, C=2.0 / l)
        for l in l_list
    }
    errs_grad_space = {
        l: _make_errors(hs_test, order=l, C=1.5 / l)
        for l in l_list
    }

    plot_spatial_convergence_all(hs_test, errs_c_space, errs_grad_space)

    # ── BDF test data ─────────────────────────────────────────────────────────
    dt_test    = np.array([1/4, 1/8, 1/16, 1/32])
    bdf_orders = [1, 2, 3, 4, 5, 6]

    errs_c_bdf = {
        k: _make_errors(dt_test, order=k, C=2.0 / k)
        for k in bdf_orders
    }
    errs_grad_bdf = {
        k: _make_errors(dt_test, order=k, C=1.5 / k)
        for k in bdf_orders
    }

    plot_time_convergence_all(dt_test, errs_c_bdf, errs_grad_bdf, method=TimeMethod.BDF)

    # ── Theta-method test data ────────────────────────────────────────────────
    # theta=0.5 (Crank-Nicolson) → order 2; theta=1.0 → order 1
    theta_values = [0.5, 1.0]

    def theta_order(theta):
        return 2 if abs(theta - 0.5) < 1e-10 else 1

    errs_c_theta = {
        th: _make_errors(dt_test, order=theta_order(th), C=2.0)
        for th in theta_values
    }
    errs_grad_theta = {
        th: _make_errors(dt_test, order=theta_order(th), C=1.5)
        for th in theta_values
    }

    plot_time_convergence_all(dt_test, errs_c_theta, errs_grad_theta, method=TimeMethod.THETA)