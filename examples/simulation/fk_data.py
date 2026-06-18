from fisher_kolmogorov.configs.plot_configs.convergence_data import ConvergenceData
from fisher_kolmogorov.utilities.enum_utilities              import PolyDegree, BdfOrder, TimeMethod

import numpy as np

# spldg_bdf ___________________________________________________________________

spldg_bdf = ConvergenceData(

    # Mesh sizes and time steps -----------------------------------------------
    hs = np.array([0.242339, 0.120878, 0.061524, 0.030154]),
    dt = np.array([0.5, 0.25, 0.125]),

    # Spatial convergence: keys are polynomial degrees ------------------------
    errs_c_space = {
        PolyDegree.P1: [1.7054e-02, 4.4260e-03, 1.0670e-03, 2.6291e-04],
        PolyDegree.P2: [2.7144e-03, 3.2527e-04, 3.6448e-05, 4.4344e-06],
        PolyDegree.P3: [2.9625e-04, 2.3229e-05, 1.4044e-06, 8.7902e-08],
        PolyDegree.P4: [5.0428e-05, 2.4900e-06, 6.8202e-08, 2.0306e-09],
        PolyDegree.P5: [1.3235e-05, 2.4196e-07, 4.0558e-09, 6.3312e-11],
        PolyDegree.P6: [2.2438e-06, 3.6816e-08, 2.5601e-10, 2.1027e-12],
        PolyDegree.P7: [4.8502e-07, 4.8345e-09, 1.7267e-11, 6.4065e-14],
        PolyDegree.P8: [1.4905e-07, 4.2447e-10, 1.0822e-12, 4.0112e-15],
    },
    errs_grad_space = {
        PolyDegree.P1: [2.9827e-01, 1.5027e-01, 7.3074e-02, 3.6567e-02],
        PolyDegree.P2: [8.8057e-02, 1.9341e-02, 3.8581e-03, 8.5960e-04],
        PolyDegree.P3: [1.3851e-02, 1.9701e-03, 2.2954e-04, 2.7575e-05],
        PolyDegree.P4: [2.8424e-03, 2.6031e-04, 1.3571e-05, 7.4553e-07],
        PolyDegree.P5: [8.8777e-04, 3.0445e-05, 9.8716e-07, 2.9406e-08],
        PolyDegree.P6: [1.7083e-04, 5.3616e-06, 7.2867e-08, 1.1214e-09],
        PolyDegree.P7: [4.3763e-05, 8.0281e-07, 5.6169e-09, 3.9918e-11],
        PolyDegree.P8: [1.4726e-05, 8.0421e-08, 4.0004e-10, 1.7951e-12],
    },

    # Polynomial convergence: one error per degree, fixed h and dt ------------
    poly_conv_h       = 0.196019,
    poly_conv_degrees = [
        PolyDegree.P1, PolyDegree.P2, PolyDegree.P3,
        PolyDegree.P4, PolyDegree.P5, PolyDegree.P6,
        PolyDegree.P7, PolyDegree.P8,
    ],
    errs_c_poly    = [7.0149e-03, 1.3491e-03, 1.2219e-04, 2.7321e-05, 7.4897e-06, 8.6814e-07, 2.6519e-07, 6.5219e-08],
    errs_grad_poly = [2.6227e-01, 6.5657e-02, 8.3423e-03, 2.0098e-03, 6.1051e-04, 7.6386e-05, 2.7662e-05, 7.2932e-06],

    # Temporal convergence: keys are BDF orders -------------------------------
    errs_c_bdf = {
        BdfOrder.BDF1: [1.6856e-01, 8.0919e-02, 3.9276e-02],
        BdfOrder.BDF2: [3.4826e-02, 8.0013e-03, 1.9731e-03],
        BdfOrder.BDF3: [7.8297e-03, 9.2030e-04, 1.1242e-04],
        BdfOrder.BDF4: [2.0493e-03, 1.1260e-04, 6.6457e-06],
        BdfOrder.BDF5: [5.5420e-04, 1.4070e-05, 3.9876e-07],
        BdfOrder.BDF6: [1.5325e-04, 1.7856e-06, 2.3991e-08],
    },
    errs_grad_bdf = {
        BdfOrder.BDF1: [5.6152e-02, 5.0004e-02, 2.9951e-02],
        BdfOrder.BDF2: [4.2004e-02, 9.8876e-03, 2.4173e-03],
        BdfOrder.BDF3: [1.1666e-02, 1.3565e-03, 1.6238e-04],
        BdfOrder.BDF4: [3.4223e-03, 1.8280e-04, 1.0593e-05],
        BdfOrder.BDF5: [9.8140e-04, 2.4256e-05, 6.7730e-07],
        BdfOrder.BDF6: [2.8145e-04, 3.2049e-06, 4.2752e-08],
    },

    # Space saturation (dt = 1e-1): fixed time method, varying BDF order ------
    space_sat_time_method = TimeMethod.BDF,
    space_sat_hs          = np.array([0.220067, 0.107795, 0.061569, 0.038996, 0.030652, 0.021536]),
    space_sat_degrees     = [PolyDegree.P2, PolyDegree.P3],
    errs_c_space_sat = {
        PolyDegree.P2: {
            BdfOrder.BDF1: [2.4854e-03, 2.5201e-03, 2.5214e-03, 2.5215e-03, 2.5216e-03, 2.5216e-03],
            BdfOrder.BDF2: [3.2386e-04, 1.2057e-04, 1.1317e-04, 1.1290e-04, 1.1287e-04, 1.1286e-04],
            BdfOrder.BDF3: [3.0880e-04, 4.4756e-05, 1.2092e-05, 9.3250e-06, 9.0872e-06, 9.0280e-06],
            BdfOrder.BDF4: [3.0698e-04, 4.3291e-05, 7.8058e-06, 2.3893e-06, 1.3732e-06, 9.9701e-07],
            BdfOrder.BDF5: [3.0707e-04, 4.3120e-05, 7.7480e-06, 2.2051e-06, 1.0163e-06, 3.7477e-07],
            BdfOrder.BDF6: [3.0601e-04, 4.3564e-05, 7.8562e-06, 2.2096e-06, 1.0103e-06, 3.5636e-07],
        },
        PolyDegree.P3: {
            BdfOrder.BDF1: [2.5227e-03, 2.5215e-03, 2.5216e-03, 2.5216e-03, 2.5216e-03, 2.5216e-03],
            BdfOrder.BDF2: [1.3304e-04, 1.1288e-04, 1.1286e-04, 1.1286e-04, 1.1286e-04, 1.1286e-04],
            BdfOrder.BDF3: [7.4100e-05, 9.4822e-06, 9.0259e-06, 9.0187e-06, 9.0185e-06, 9.0184e-06],
            BdfOrder.BDF4: [7.4215e-05, 2.8250e-06, 9.8588e-07, 9.3269e-07, 9.3142e-07, 9.3116e-07],
            BdfOrder.BDF5: [7.4362e-05, 2.6363e-06, 3.4043e-07, 1.3024e-07, 1.2076e-07, 1.1902e-07],
            BdfOrder.BDF6: [7.4611e-05, 2.6563e-06, 3.2399e-07, 5.7441e-08, 2.9175e-08, 2.0857e-08],
        },
    },
    errs_grad_space_sat = {
        PolyDegree.P2: {
            BdfOrder.BDF1: [4.2327e-02, 3.0015e-02, 2.9622e-02, 2.9605e-02, 2.9603e-02, 2.9602e-02],
            BdfOrder.BDF2: [3.2653e-02, 5.6845e-03, 2.2661e-03, 1.9447e-03, 1.9098e-03, 1.8998e-03],
            BdfOrder.BDF3: [3.2652e-02, 5.3461e-03, 1.2554e-03, 4.6674e-04, 2.8769e-04, 2.1074e-04],
            BdfOrder.BDF4: [3.2657e-02, 5.3391e-03, 1.2415e-03, 4.2891e-04, 2.2068e-04, 1.0171e-04],
            BdfOrder.BDF5: [3.2654e-02, 5.3370e-03, 1.2429e-03, 4.2858e-04, 2.1954e-04, 9.9196e-05],
            BdfOrder.BDF6: [3.2640e-02, 5.3772e-03, 1.2498e-03, 4.2883e-04, 2.1929e-04, 9.9092e-05],
        },
        PolyDegree.P3: {
            BdfOrder.BDF1: [3.1078e-02, 2.9607e-02, 2.9602e-02, 2.9602e-02, 2.9602e-02, 2.9602e-02],
            BdfOrder.BDF2: [1.0762e-02, 2.0097e-03, 1.8994e-03, 1.8973e-03, 1.8972e-03, 1.8972e-03],
            BdfOrder.BDF3: [1.0569e-02, 6.8997e-04, 2.0724e-04, 1.8700e-04, 1.8619e-04, 1.8604e-04],
            BdfOrder.BDF4: [1.0555e-02, 6.6404e-04, 9.4082e-05, 2.9709e-05, 2.4104e-05, 2.2875e-05],
            BdfOrder.BDF5: [1.0554e-02, 6.6343e-04, 9.1349e-05, 1.9406e-05, 8.6493e-06, 4.1330e-06],
            BdfOrder.BDF6: [1.0561e-02, 6.6348e-04, 9.1391e-05, 1.9136e-05, 8.0032e-06, 2.5196e-06],
        },
    },

    # Polynomial saturation (negative history init): fixed h, degree on x-axis
    poly_sat_h           = 0.134894,
    poly_sat_degrees     = [
        PolyDegree.P1, PolyDegree.P2, PolyDegree.P3,
        PolyDegree.P4, PolyDegree.P5, PolyDegree.P6,
    ],
    poly_sat_time_method = TimeMethod.BDF,
    errs_c_poly_sat = {
        BdfOrder.BDF1: [1.4591e-02, 1.4425e-02, 1.4318e-02, 1.4317e-02, 1.4317e-02, 1.4317e-02],
        BdfOrder.BDF2: [1.7090e-03, 3.8889e-04, 2.5275e-04, 2.5208e-04, 2.5210e-04, 2.5210e-04],
        BdfOrder.BDF3: [1.6854e-03, 1.8592e-04, 1.2429e-05, 5.1385e-06, 4.9552e-06, 4.9521e-06],
        BdfOrder.BDF4: [1.6854e-03, 1.8296e-04, 1.1250e-05, 1.4688e-06, 2.0761e-07, 1.0648e-07],
        BdfOrder.BDF5: [1.6854e-03, 1.8290e-04, 1.1247e-05, 1.4671e-06, 1.8097e-07, 3.1230e-08],
        BdfOrder.BDF6: [1.6854e-03, 1.8290e-04, 1.1247e-05, 1.4672e-06, 1.8090e-07, 3.0797e-08],
    },
    errs_grad_poly_sat = {
        BdfOrder.BDF1: [1.3784e-01, 1.0550e-01, 1.0484e-01, 1.0482e-01, 1.0482e-01, 1.0482e-01],
        BdfOrder.BDF2: [8.8577e-02, 1.1238e-02, 3.3476e-03, 1.8404e-03, 1.8119e-03, 1.8100e-03],
        BdfOrder.BDF3: [8.8605e-02, 1.1044e-02, 2.8275e-03, 3.4052e-04, 9.2975e-05, 3.7176e-05],
        BdfOrder.BDF4: [8.8607e-02, 1.1043e-02, 2.8274e-03, 3.3888e-04, 8.6432e-05, 1.4327e-05],
        BdfOrder.BDF5: [8.8607e-02, 1.1043e-02, 2.8274e-03, 3.3888e-04, 8.6430e-05, 1.4312e-05],
        BdfOrder.BDF6: [8.8607e-02, 1.1043e-02, 2.8274e-03, 3.3888e-04, 8.6430e-05, 1.4312e-05],
    },
)