from Plots.PlotUtilities      import *
from Utilities.EnumUtilities  import PolyDegree, BdfOrder, ThetaMethod, TimeMethod

import numpy as np

if __name__ == "__main__":

    rng = np.random.default_rng(0)

    def _make_errors(xs, order, C=1.0, saturate_at=None):
        """
        Errors decaying like C * xs^order with mild noise.
        If saturate_at is given, errors flatten below that floor
        (simulates temporal discretisation saturation).
        """
        raw = C * xs**order * (1 + 0.04 * rng.standard_normal(len(xs)))
        if saturate_at is not None:
            raw = np.maximum(raw, saturate_at)
        return raw

    # ── Spatial convergence ───────────────────────────────────────────────────
    hs_test = np.array([1/4, 1/8, 1/16, 1/32])

    errs_c_space = {
        l: _make_errors(hs_test, order=int(l) + 1, C=2.0 / int(l))
        for l in PolyDegree
    }
    errs_grad_space = {
        l: _make_errors(hs_test, order=int(l), C=1.5 / int(l))
        for l in PolyDegree
    }

    plot_spatial_convergence_all(hs_test, errs_c_space, errs_grad_space)

    # ── BDF temporal convergence ──────────────────────────────────────────────
    dt_test = np.array([1/2, 1/4, 1/8, 1/16])*0.25

    errs_c_bdf = {
        k: _make_errors(dt_test, order=int(k), C=2.0 / int(k))
        for k in BdfOrder
    }
    errs_grad_bdf = {
        k: _make_errors(dt_test, order=int(k), C=1.5 / int(k))
        for k in BdfOrder
    }

    plot_time_convergence_all(dt_test, errs_c_bdf, errs_grad_bdf, method=TimeMethod.BDF)

    # ── Theta temporal convergence ────────────────────────────────────────────
    _theta_order = {ThetaMethod.CN: 2, ThetaMethod.IE: 1}

    errs_c_theta = {
        th: _make_errors(dt_test, order=_theta_order[th], C=2.0)
        for th in (ThetaMethod.CN, ThetaMethod.IE)
    }
    errs_grad_theta = {
        th: _make_errors(dt_test, order=_theta_order[th], C=1.5)
        for th in (ThetaMethod.CN, ThetaMethod.IE)
    }

    plot_time_convergence_all(dt_test, errs_c_theta, errs_grad_theta, method=TimeMethod.THETA)

    # ── Test slope triangle with WRONG data ──────────────────────────────────
    errs_wrong = {
        l: _make_errors(hs_test, order=1, C=2.0 / l)
        for l in [1, 2, 3]
    }
    errs_grad_wrong = {
        l: _make_errors(hs_test, order=1, C=1.5 / l)
        for l in [1, 2, 3]
    }

    plot_spatial_convergence_all(hs_test, errs_wrong, errs_grad_wrong)

    # # ── Spatial saturation study ──────────────────────────────────────────────
    # l_sat = PolyDegree.P3
    # hs_sat = np.array([1/4, 1/8, 1/16, 1/32, 1/64, 1/128])

    # def _spatial_floor(nu: BdfOrder, hs, l: PolyDegree, C=2.0):
    #     """
    #     Floor that intersects the spatial curve C*h^(l+1) at hs[-(7 - int(nu))],
    #     so BDF6 never saturates, BDF5 saturates only the last point, etc.
    #     BDF1 -> saturates last 5 points, BDF6 -> saturates 0 points.
    #     """
    #     sat_index = 7 - int(nu)          # index from the right where saturation begins
    #     if sat_index >= len(hs):
    #         return 0.0                   # BDF6: floor below all points → no saturation
    #     return C * hs[-(sat_index)] ** (int(l) + 1) * 0.9

    # def _grad_floor(nu: BdfOrder, hs, l: PolyDegree, C=1.5):
    #     sat_index = 7 - int(nu)
    #     if sat_index >= len(hs):
    #         return 0.0
    #     return C * hs[-(sat_index)] ** int(l) * 0.9

    # errs_c_sat = {
    #     nu: np.maximum(
    #         2.0 * hs_sat ** (int(l_sat) + 1) * (1 + 0.04 * rng.standard_normal(len(hs_sat))),
    #         _spatial_floor(nu, hs_sat, l_sat, C=2.0),
    #     )
    #     for nu in BdfOrder
    # }
    # errs_grad_sat = {
    #     nu: np.maximum(
    #         1.5 * hs_sat ** int(l_sat) * (1 + 0.04 * rng.standard_normal(len(hs_sat))),
    #         _grad_floor(nu, hs_sat, l_sat, C=1.5),
    #     )
    #     for nu in BdfOrder
    # }

    # plot_spatial_saturation(hs_sat, errs_c_sat, errs_grad_sat, l=l_sat, method=TimeMethod.BDF)


    # # ── Spatial saturation study — Theta method ───────────────────────────────
    # errs_c_sat_theta = {
    #     th: _make_errors(
    #         hs_sat,
    #         order  = int(l_sat) + 1,
    #         C      = 2.0,
    #         saturate_at = 0.8 * dt_fixed ** _theta_order[th],
    #     )
    #     for th in (ThetaMethod.CN, ThetaMethod.IE)
    # }
    # errs_grad_sat_theta = {
    #     th: _make_errors(
    #         hs_sat,
    #         order  = int(l_sat),
    #         C      = 1.5,
    #         saturate_at = 0.6 * dt_fixed ** _theta_order[th],
    #     )
    #     for th in (ThetaMethod.CN, ThetaMethod.IE)
    # }

    # plot_spatial_saturation(hs_sat, errs_c_sat_theta, errs_grad_sat_theta,
    #                         l=l_sat, method=TimeMethod.THETA)
    
    # # ── Polynomial saturation study — BDF ────────────────────────────────────
    # # Fixed h=1/8, l from 1 to 6. Floor ~ dt_fixed^nu as before.
    # h_sat  = 1/8
    # l_list_sat = list(PolyDegree)[:6]   # P1..P6

    # errs_c_poly_sat = {
    #     nu: _make_errors(
    #         h_sat ** np.array(l_ints_sat := [int(l) + 1 for l in l_list_sat]),
    #         order = 1,   # already encoded in the exponent above
    #         C     = 1.0,
    #         saturate_at = 0.8 * dt_fixed ** int(nu),
    #     )
    #     for nu in BdfOrder
    # }