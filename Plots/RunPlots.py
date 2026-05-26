from Utilities.EnumUtilities import ConvType, TimeMethod, SpaceMethod, StudyType, PolyDegree, BdfOrder, ThetaMethod

# Plot selector ___________________________________________________________________________________________________________________________
PLOT          = ConvType.POLYNOMIAL
TIME          = TimeMethod.BDF
SPACE         = SpaceMethod.SPLDG
STUDY         = StudyType.SATURATION
SpaceSatOrder = PolyDegree.P3
COMBINED      = True
SAVE          = True

# Dynamic config import ___________________________________________________________________________________________________________________
if SPACE == SpaceMethod.SPLDG:
    if TIME == TimeMethod.BDF:
        from Plots.ModelConfig.SPLDG_BDF   import *
    elif TIME == TimeMethod.THETA:
        from Plots.ModelConfig.SPLDG_THETA import *
    else:
        raise ValueError(f"Unsupported time method '{TIME}' for SPLDG.")
elif SPACE == SpaceMethod.PPDG:
    if TIME == TimeMethod.BDF:
        from Plots.ModelConfig.PPDG_BDF    import *
    elif TIME == TimeMethod.THETA:
        from Plots.ModelConfig.PPDG_THETA  import *
    else:
        raise ValueError(f"Unsupported time method '{TIME}' for PPDG.")
elif SPACE == SpaceMethod.LDG:
    if TIME == TimeMethod.BDF:
        from Plots.ModelConfig.LDG_BDF    import *
    elif TIME == TimeMethod.THETA:
        from Plots.ModelConfig.LDG_THETA  import *
    else:
        raise ValueError(f"Unsupported time method '{TIME}' for LDG.")
elif SPACE == SpaceMethod.DG:
    if TIME == TimeMethod.BDF:
        from Plots.ModelConfig.DG_BDF    import *
    elif TIME == TimeMethod.THETA:
        from Plots.ModelConfig.DG_THETA  import *
    else:
        raise ValueError(f"Unsupported time method '{TIME}' for DG.")
else:
    raise ValueError(f"Unsupported space method '{SPACE}'.")

from Plots.PlotUtilities import (
    plot_spatial_convergence_all,
    plot_time_convergence_all,
    plot_polynomial_convergence,
    plot_spatial_saturation,
    plot_polynomial_saturation,
    plot_combined_poly_and_time,
    plot_spatial_saturation_combined,
)

# Runtime dispatch ________________________________________________________________________________________________________________________
if __name__ == "__main__":

    if STUDY == StudyType.CONVERGENCE:

        if PLOT == ConvType.SPATIAL:
            if TIME == TimeMethod.THETA:
                raise ValueError("Spatial convergence data not available for THETA method.")
            plot_spatial_convergence_all(hs, errs_c_space, errs_grad_space, space_method=SPACE, time_method=TIME, save=SAVE)

        elif PLOT == ConvType.POLYNOMIAL:
            if TIME == TimeMethod.THETA:
                raise ValueError("Polynomial convergence data not available for THETA method.")
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
                plot_time_convergence_all(dt, errs_c_bdf,   errs_grad_bdf,   time_method=TimeMethod.BDF,   space_method=SPACE, save=SAVE)
            elif TIME == TimeMethod.THETA:
                plot_time_convergence_all(dt, errs_c_theta, errs_grad_theta, time_method=TimeMethod.THETA, space_method=SPACE, save=SAVE)

        else:
            raise ValueError(f"Unknown plot type '{PLOT}'. Choose from: {list(ConvType)}.")

    elif STUDY == StudyType.SATURATION:

        if PLOT == ConvType.SPATIAL:
            if TIME == TimeMethod.THETA:
                raise ValueError("Spatial saturation data not available for THETA method.")
            if COMBINED:
                plot_spatial_saturation_combined(
                    SpaceSatHs,
                    errs_c_space_sat_by_degree,
                    errs_grad_space_sat_by_degree,
                    degrees      = SpaceSatDegrees,
                    time_method  = SpaceSatTimeMethod,
                    space_method = SPACE,
                    save         = SAVE,
                )
            else:
                errs_c_space_sat    = errs_c_space_sat_by_degree[SpaceSatOrder]
                errs_grad_space_sat = errs_grad_space_sat_by_degree[SpaceSatOrder]
                plot_spatial_saturation(
                    SpaceSatHs, errs_c_space_sat, errs_grad_space_sat,
                    l=SpaceSatOrder, time_method=SpaceSatTimeMethod,
                    space_method=SPACE, save=SAVE,
                )

        elif PLOT == ConvType.POLYNOMIAL:
            if TIME == TimeMethod.THETA:
                raise ValueError("Polynomial saturation data not available for THETA method.")
            plot_polynomial_saturation(
                PolySatLList, errs_c_poly_sat, errs_grad_poly_sat,
                h=PolySatH, time_method=PolySatTimeMethod, space_method=SPACE,
                save=SAVE,
            )

        elif PLOT == ConvType.TEMPORAL:
            raise ValueError(f"Saturation study not supported for '{PLOT}'. Choose from: {list([ConvType.SPATIAL, ConvType.POLYNOMIAL])}.")