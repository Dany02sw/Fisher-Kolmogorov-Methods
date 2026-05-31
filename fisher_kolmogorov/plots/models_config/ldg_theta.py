from fisher_kolmogorov.utilities.enum_utilities import TimeMethod, ThetaMethod, PolyDegree

import numpy as np

# Data to fill in manually ________________________________________________________________________________________________________________
hs = np.array([...]) # mesh sizes used for spatial convergence
dt = np.array([0.5, 0.25, 0.125]) # time steps used for temporal convergence


# Spatial convergence: keys are polynomial degrees ----------------------------------------------------------------------------------------
errs_c_space = {
    PolyDegree.P1: [...],
    PolyDegree.P2: [...],
    PolyDegree.P3: [...],
    PolyDegree.P4: [...],
}
errs_grad_space = {
    PolyDegree.P1: [...],
    PolyDegree.P2: [...],
    PolyDegree.P3: [...],
    PolyDegree.P4: [...],
}


# Polynomial convergence: one error per degree, fixed h and dt ----------------------------------------------------------------------------
PolyConvH      = ...
PolyConvLList  = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3,
                  PolyDegree.P4, PolyDegree.P5, PolyDegree.P6,
                  PolyDegree.P7, PolyDegree.P8]
errs_c_poly    = [...]   # one value per degree in PolyConvLList
errs_grad_poly = [...]


# Theta convergence: keys are ThetaMethod values ------------------------------------------------------------------------------------------
errs_c_theta = {
    ThetaMethod.CN: [3.8805e-03, 9.7175e-04, 2.4348e-04],
    ThetaMethod.IE: [2.6821e-01, 1.3710e-01, 6.7986e-02],
}
errs_grad_theta = {
    ThetaMethod.CN: [3.2627e-02, 8.4431e-03, 2.1295e-03],
    ThetaMethod.IE: [1.2761e-01, 1.0343e-01, 6.1795e-02],
}

# Space saturation: fixed poly degree, varying time order ----------------------------------------------------------------------------------
SpaceSatTimeMethod = TimeMethod.THETA
SpaceSatHs         = np.array([0.107795, 0.071439, 0.047590, 0.030652, 0.021536])
SpaceSatDegrees    = [PolyDegree.P2, PolyDegree.P3]

errs_c_space_sat_by_degree = {
    PolyDegree.P2: {
        ThetaMethod.CN: [...],
        ThetaMethod.IE: [...],
    },
    PolyDegree.P3: {
        ThetaMethod.CN: [...],
        ThetaMethod.IE: [...],
    },
}
errs_grad_space_sat_by_degree = {
    PolyDegree.P2: {
        ThetaMethod.CN: [...],
        ThetaMethod.IE: [...],
    },
    PolyDegree.P3: {
        ThetaMethod.CN: [...],
        ThetaMethod.IE: [...],
    },
}

# Polynomial saturation: fixed h, varying time order, degree on x-axis -----------------------------------------------------------------------
PolySatH           = 0.107795
PolySatLList       = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3,
                      PolyDegree.P4, PolyDegree.P5, PolyDegree.P6]
PolySatTimeMethod  = TimeMethod.THETA

errs_c_poly_sat = {
    ThetaMethod.CN: [...],
    ThetaMethod.IE: [...],
}
errs_grad_poly_sat = {
    ThetaMethod.CN: [...],
    ThetaMethod.IE: [...],
}
