from Utilities.PlotUtilities import plot_spatial_convergence_all, plot_time_convergence_all
from Utilities.EnumUtilities import ConvType, TimeMethod, SpaceMethod
import numpy as np

# ── Data to fill in manually ──────────────────────────────────────────────────

hs = np.array([1/4, 1/8, 1/16, 1/32])
dt = np.array([0.5, 0.25, 0.125])

# Spatial convergence: keys are polynomial degrees
errs_c_space = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}
errs_grad_space = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}

# BDF convergence: keys are BDF orders
errs_c_bdf = {
    1: [2.6821e-01, 1.3710e-01, 6.7986e-02],
    2: [1.0375e-02, 2.5226e-03, 6.3198e-04],
    3: [2.0679e-02, 2.3372e-03, 2.8002e-04],
    4: [6.7201e-03, 3.4099e-04, 1.9436e-05],
    5: [2.2266e-03, 5.0772e-05, 1.3678e-06],
    6: [7.6250e-04, 7.6999e-06, 9.6438e-08],
}
errs_grad_bdf = {
    1: [1.2762e-01, 1.0343e-01, 6.1796e-02],
    2: [9.6063e-03, 2.4729e-03, 6.1440e-04],
    3: [3.3741e-02, 3.8215e-03, 4.5170e-04],
    4: [1.2087e-02, 5.9996e-04, 3.3755e-05],
    5: [4.1492e-03, 9.2967e-05, 2.4787e-06],
    6: [1.4546e-03, 1.4443e-05, 1.7953e-07],
}

# Theta convergence: keys are theta values (0.5 or 1.0)
errs_c_theta = {
    0.5: [],
    1.0: [],
}
errs_grad_theta = {
    0.5: [],
    1.0: [],
}

# ── Plot selector ─────────────────────────────────────────────────────────────
PLOT  = ConvType.TEMPORAL
TIME  = TimeMethod.BDF
SPACE = SpaceMethod.SPLDG

# ── Runtime dispatch ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    if PLOT == ConvType.SPATIAL:
        plot_spatial_convergence_all(hs, errs_c_space, errs_grad_space, method=SPACE)

    elif PLOT == ConvType.TEMPORAL:

        if TIME == TimeMethod.BDF:
            plot_time_convergence_all(dt, errs_c_bdf, errs_grad_bdf, method=TimeMethod.BDF, space_method=SPACE)

        elif TIME == TimeMethod.THETA:
            plot_time_convergence_all(dt, errs_c_theta, errs_grad_theta, method=TimeMethod.THETA, space_method=SPACE)

    else:
        raise ValueError(f"Unknown plot type '{PLOT}'. Choose from: {[ConvType.SPATIAL.value, ConvType.TEMPORAL.value]}.")