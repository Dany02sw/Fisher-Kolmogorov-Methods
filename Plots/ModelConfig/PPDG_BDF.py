from Utilities.EnumUtilities import PolyDegree, TimeMethod, BdfOrder
import numpy as np

# Data to fill in manually ________________________________________________________________________________________________________________
hs = np.array([...])
dt = np.array([...])

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

# BDF convergence: keys are BDF orders ----------------------------------------------------------------------------------------------------
errs_c_bdf = {
    BdfOrder.BDF1: [...],
    BdfOrder.BDF2: [...],
    BdfOrder.BDF3: [...],
    BdfOrder.BDF4: [...],
    BdfOrder.BDF5: [...],
    BdfOrder.BDF6: [...],
}
errs_grad_bdf = {
    BdfOrder.BDF1: [...],
    BdfOrder.BDF2: [...],
    BdfOrder.BDF3: [...],
    BdfOrder.BDF4: [...],
    BdfOrder.BDF5: [...],
    BdfOrder.BDF6: [...],
}

# Space saturation: fixed poly degree, varying time order ---------------------------------------------------------------------------------
SpaceSatTimeMethod = TimeMethod.BDF
SpaceSatHs         = np.array([...])
SpaceSatDegrees    = [PolyDegree.P2, PolyDegree.P3]

errs_c_space_sat_by_degree = {
    PolyDegree.P2: {
        BdfOrder.BDF1: [...],
        BdfOrder.BDF2: [...],
        BdfOrder.BDF3: [...],
        BdfOrder.BDF4: [...],
        BdfOrder.BDF5: [...],
        BdfOrder.BDF6: [...],
    },
    PolyDegree.P3: {
        BdfOrder.BDF1: [...],
        BdfOrder.BDF2: [...],
        BdfOrder.BDF3: [...],
        BdfOrder.BDF4: [...],
        BdfOrder.BDF5: [...],
        BdfOrder.BDF6: [...],
    },
}
errs_grad_space_sat_by_degree = {
    PolyDegree.P2: {
        BdfOrder.BDF1: [...],
        BdfOrder.BDF2: [...],
        BdfOrder.BDF3: [...],
        BdfOrder.BDF4: [...],
        BdfOrder.BDF5: [...],
        BdfOrder.BDF6: [...],
    },
    PolyDegree.P3: {
        BdfOrder.BDF1: [...],
        BdfOrder.BDF2: [...],
        BdfOrder.BDF3: [...],
        BdfOrder.BDF4: [...],
        BdfOrder.BDF5: [...],
        BdfOrder.BDF6: [...],
    },
}

# Polynomial saturation: fixed h, varying time order, degree on x-axis -------------------------------------------------------------------
PolySatH          = ...
PolySatLList      = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3,
                     PolyDegree.P4, PolyDegree.P5, PolyDegree.P6]
PolySatTimeMethod = TimeMethod.BDF

errs_c_poly_sat = {
    BdfOrder.BDF1: [...],
    BdfOrder.BDF2: [...],
    BdfOrder.BDF3: [...],
    BdfOrder.BDF4: [...],
    BdfOrder.BDF5: [...],
    BdfOrder.BDF6: [...],
}
errs_grad_poly_sat = {
    BdfOrder.BDF1: [...],
    BdfOrder.BDF2: [...],
    BdfOrder.BDF3: [...],
    BdfOrder.BDF4: [...],
    BdfOrder.BDF5: [...],
    BdfOrder.BDF6: [...],
}