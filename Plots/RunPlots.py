from Plots.PlotUtilities import (
    plot_spatial_convergence_all,
    plot_time_convergence_all,
    plot_polynomial_convergence,
    plot_spatial_saturation,
    plot_polynomial_saturation,
    plot_combined_poly_and_time
)
from Utilities.EnumUtilities import ConvType, TimeMethod, SpaceMethod, StudyType, PolyDegree, BdfOrder, ThetaMethod
import numpy as np

# Data to fill in manually ________________________________________________________________________________________________________________
hs = np.array([0.242339, 0.120878, 0.061524, 0.030154])
dt = np.array([0.5, 0.25, 0.125])

# Spatial convergence: keys are polynomial degrees ----------------------------------------------------------------------------------------
errs_c_space = {
    PolyDegree.P1: [1.3719e-02, 3.5956e-03, 8.9795e-04, 2.2890e-04],
    PolyDegree.P2: [2.8470e-03, 3.6232e-04, 4.0171e-05, 4.9172e-06],
    PolyDegree.P3: [3.4550e-04, 2.5007e-05, 1.5871e-06, 1.0034e-07],
    PolyDegree.P4: [5.5571e-05, 3.2943e-06, 8.5627e-08, 2.5341e-09],
}
errs_grad_space = {
    PolyDegree.P1: [3.2122e-01, 1.6312e-01, 7.9042e-02, 3.9483e-02],
    PolyDegree.P2: [9.7128e-02, 2.1496e-02, 4.2787e-03, 9.5171e-04],
    PolyDegree.P3: [1.5790e-02, 2.1195e-03, 2.5958e-04, 3.1329e-05],
    PolyDegree.P4: [3.1264e-03, 3.4274e-04, 1.6988e-05, 9.2847e-07],
}

# Polynomial convergence: one error per degree, fixed h and dt ----------------------------------------------------------------------------
PolyConvH      = 0.196019
PolyConvLList  = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3,
                  PolyDegree.P4, PolyDegree.P5, PolyDegree.P6,
                  PolyDegree.P7, PolyDegree.P8]
errs_c_poly    = [7.0142e-03, 1.3494e-03, 1.2236e-04, 2.7339e-05, 7.4957e-06, 8.6946e-07, 2.6554e-07, 6.5268e-08]   # one value per degree in PolyConvLList
errs_grad_poly = [2.6228e-01, 6.5734e-02, 8.3562e-03, 2.0082e-03, 6.0901e-04, 7.6389e-05, 2.7612e-05, 7.3085e-06]

# BDF convergence: keys are BDF orders -----------------------------------------------------------------------------------------------------
errs_c_bdf = {
    BdfOrder.BDF1: [1.6856e-01, 8.0923e-02, 3.9280e-02],
    BdfOrder.BDF2: [3.4818e-02, 7.9948e-03, 1.9668e-03],
    BdfOrder.BDF3: [7.8376e-03, 9.2873e-04, 1.2118e-04],
    BdfOrder.BDF4: [2.0493e-03, 1.1260e-04, 6.6457e-06],
    BdfOrder.BDF5: [5.5420e-04, 1.4070e-05, 3.9876e-07],
    BdfOrder.BDF6: [1.5325e-04, 1.7856e-06, 2.3991e-08],
}
errs_grad_bdf = {
    BdfOrder.BDF1: [5.6138e-02, 4.9987e-02, 2.9934e-02],
    BdfOrder.BDF2: [4.2022e-02, 9.9057e-03, 2.4406e-03],
    BdfOrder.BDF3: [1.1651e-02, 1.3410e-03, 1.4755e-04],
    BdfOrder.BDF4: [3.4223e-03, 1.8280e-04, 1.0593e-05],
    BdfOrder.BDF5: [9.8140e-04, 2.4256e-05, 6.7742e-07],
    BdfOrder.BDF6: [2.8145e-04, 3.2049e-06, 4.3734e-08],
}

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
SpaceSatTimeMethod = TimeMethod.BDF
SpaceSatHs = np.array([0.107795, 0.071439, 0.047590, 0.030652, 0.021536])

# l = 2
SpaceSatOrder    = PolyDegree.P2
errs_c_space_sat = {
    BdfOrder.BDF1: [4.608814e-04, 1.775473e-04, 9.038723e-05, 7.162949e-05, 6.999359e-05],
    BdfOrder.BDF2: [4.536665e-04, 1.635479e-04, 5.766381e-05, 1.651438e-05, 5.757205e-06],
    BdfOrder.BDF3: [4.509979e-04, 1.630636e-04, 5.761942e-05, 1.651403e-05, 5.752545e-06],
    BdfOrder.BDF4: [4.481829e-04, 1.625309e-04, 5.755877e-05, 1.650767e-05, 5.751708e-06],
    BdfOrder.BDF5: [4.452452e-04, 1.619626e-04, 5.749407e-05, 1.650110e-05, 5.750582e-06],
    BdfOrder.BDF6: [4.425511e-04, 1.615723e-04, 5.751075e-05, 1.639373e-05, 5.769531e-06],
}
errs_grad_space_sat = {
    BdfOrder.BDF1: [1.229680e-02, 3.367459e-03, 1.391055e-03, 9.397654e-04, 8.955685e-04],
    BdfOrder.BDF2: [1.245412e-02, 3.295150e-03, 1.078848e-03, 3.028853e-04, 1.248636e-04],
    BdfOrder.BDF3: [1.260931e-02, 3.338545e-03, 1.083578e-03, 3.033418e-04, 1.248998e-04],
    BdfOrder.BDF4: [1.277644e-02, 3.386035e-03, 1.088181e-03, 3.037396e-04, 1.249322e-04],
    BdfOrder.BDF5: [1.295257e-02, 3.437880e-03, 1.093648e-03, 3.040395e-04, 1.249884e-04],
    BdfOrder.BDF6: [1.316300e-02, 3.526121e-03, 1.107194e-03, 3.021296e-04, 1.459843e-04],
}

# l = 3
SpaceSatOrder    = PolyDegree.P3
errs_c_space_sat = {
    BdfOrder.BDF1: [8.672828e-05, 7.072806e-05, 6.982922e-05, 6.976555e-05, 6.975881e-05],
    BdfOrder.BDF2: [5.042130e-05, 1.044042e-05, 2.527967e-06, 6.051525e-07, 3.936901e-07],
    BdfOrder.BDF3: [5.022734e-05, 1.036740e-05, 2.443428e-06, 4.297334e-07, 1.101742e-07],
    BdfOrder.BDF4: [5.006667e-05, 1.034842e-05, 2.439869e-06, 4.292007e-07, 1.100969e-07],
    BdfOrder.BDF5: [4.989803e-05, 1.032854e-05, 2.436201e-06, 4.286456e-07, 1.099550e-07],
    BdfOrder.BDF6: [4.971504e-05, 1.029468e-05, 2.436129e-06, 4.270183e-07, 1.095066e-07],
}
errs_grad_space_sat = {
    BdfOrder.BDF1: [1.299393e-03, 9.199373e-04, 8.905279e-04, 8.882805e-04, 8.875738e-04],
    BdfOrder.BDF2: [1.002005e-03, 2.243346e-04, 5.613158e-05, 1.376259e-05, 7.318533e-06],
    BdfOrder.BDF3: [1.005321e-03, 2.238106e-04, 5.491158e-05, 1.142520e-05, 3.053523e-06],
    BdfOrder.BDF4: [1.009510e-03, 2.243566e-04, 5.497476e-05, 1.143001e-05, 3.055361e-06],
    BdfOrder.BDF5: [1.014097e-03, 2.249262e-04, 5.505524e-05, 1.143418e-05, 3.054173e-06],
    BdfOrder.BDF6: [1.018886e-03, 2.280204e-04, 5.780452e-05, 1.151793e-05, 3.156305e-06],
}

# Polynomial saturation: fixed h, varying time order, degree on x-axis -----------------------------------------------------------------------
PolySatH           = 0.107795
PolySatLList       = [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3,
                      PolyDegree.P4, PolyDegree.P5, PolyDegree.P6]
PolySatTimeMethod  = TimeMethod.BDF

errs_c_poly_sat = {
    BdfOrder.BDF1: [4.159532e-02, 1.269997e-02, 1.459502e-02, 1.430238e-02, 1.431912e-02, 1.431724e-02],
    BdfOrder.BDF2: [2.867489e-02, 1.631491e-03, 5.348742e-04, 2.363928e-04, 2.528155e-04, 2.509405e-04],
    BdfOrder.BDF3: [2.846105e-02, 1.846630e-03, 2.932354e-04, 1.472049e-05, 6.747140e-06, 4.851388e-06],
    BdfOrder.BDF4: [2.845421e-02, 1.849567e-03, 2.880453e-04, 1.820152e-05, 2.027122e-06, 1.704483e-07],
    BdfOrder.BDF5: [2.845092e-02, 1.848252e-03, 2.873480e-04, 1.819071e-05, 1.926091e-06, 1.661534e-07],
    BdfOrder.BDF6: [2.844735e-02, 1.846805e-03, 2.867249e-04, 1.810733e-05, 1.909231e-06, 1.659816e-07],
}
errs_grad_poly_sat = {
    BdfOrder.BDF1: [2.886474e-01, 9.320903e-02, 1.068309e-01, 1.047137e-01, 1.048314e-01, 1.048187e-01],
    BdfOrder.BDF2: [2.121636e-01, 1.436679e-02, 4.131658e-03, 1.698486e-03, 1.815034e-03, 1.802098e-03],
    BdfOrder.BDF3: [2.111351e-01, 1.570218e-02, 2.566717e-03, 1.413754e-04, 5.417567e-05, 3.381929e-05],
    BdfOrder.BDF4: [2.110957e-01, 1.571906e-02, 2.537089e-03, 1.609875e-04, 3.119533e-05, 3.353451e-06],
    BdfOrder.BDF5: [2.110713e-01, 1.571022e-02, 2.533387e-03, 1.609734e-04, 3.088498e-05, 3.327305e-06],
    BdfOrder.BDF6: [2.110450e-01, 1.570053e-02, 2.530134e-03, 1.605275e-04, 3.082610e-05, 3.327441e-06],
}


# Plot selector ___________________________________________________________________________________________________________________________
PLOT     = ConvType.SPATIAL
TIME     = TimeMethod.BDF
SPACE    = SpaceMethod.SPLDG
STUDY    = StudyType.SATURATION
COMBINED = True
SAVE     = True

# Runtime dispatch ________________________________________________________________________________________________________________________
if __name__ == "__main__":

    if STUDY == StudyType.CONVERGENCE:

        if PLOT == ConvType.SPATIAL:
            plot_spatial_convergence_all(hs, errs_c_space, errs_grad_space, space_method=SPACE, time_method=TIME, save=SAVE)

        elif PLOT == ConvType.POLYNOMIAL:
            
            if COMBINED:
                plot_combined_poly_and_time(
                    errors_c_poly    = errs_c_poly,
                    errors_grad_poly = errs_grad_poly,
                    l_list           = PolyConvLList,
                    h                = PolyConvH,
                    dt_list          = dt,
                    errs_c_time      = errs_c_bdf,
                    time_method      = TIME,
                    space_method     = SPACE,
                    save             = SAVE,
                )
            else:
                plot_polynomial_convergence(errs_c_poly, errs_grad_poly, PolyConvLList, PolyConvH, method=SPACE)

        elif PLOT == ConvType.TEMPORAL:

            if TIME == TimeMethod.BDF:
                plot_time_convergence_all(dt, errs_c_bdf, errs_grad_bdf, time_method=TimeMethod.BDF, space_method=SPACE, save=SAVE)

            elif TIME == TimeMethod.THETA:
                plot_time_convergence_all(dt, errs_c_theta, errs_grad_theta, time_method=TimeMethod.THETA, space_method=SPACE, save=SAVE)

            else:
                raise ValueError(f"Unknown time method '{TIME}'.")

        else:
            raise ValueError(f"Unknown plot type '{PLOT}'. Choose from: {list(ConvType)}.")

    elif STUDY == StudyType.SATURATION:

        if PLOT == ConvType.SPATIAL:
            plot_spatial_saturation(
                SpaceSatHs, errs_c_space_sat, errs_grad_space_sat,
                l=SpaceSatOrder, time_method=SpaceSatTimeMethod, space_method=SPACE,
                save=SAVE
            )

        elif PLOT == ConvType.POLYNOMIAL:
            plot_polynomial_saturation(
                PolySatLList, errs_c_poly_sat, errs_grad_poly_sat,
                h=PolySatH, time_method=PolySatTimeMethod, space_method=SPACE,
                save=SAVE
            )
        elif PLOT == ConvType.TEMPORAL:
            raise ValueError(f"Saturation study not supported for '{PLOT}'. Choose from: {list([ConvType.SPATIAL, ConvType.POLYNOMIAL])}.")