from SolverDgBDF import SolverDgBDF

from dolfin import *

from Meshes.Meshes                import mesh_factory
from Utilities.ProfilingUtilities import timer
from Utilities.EnumUtilities      import ConvType, MeshType, MeshStructure
from Plots.PlotUtilities          import plot_spatial_convergence, plot_polynomial_convergence, plot_time_convergence
from Utilities.PrintUtilities     import print_space_rates, print_polynomial_rates, print_time_rates, print_title, print_subtitle
from Utilities.MathUtilities      import get_decimals

if __name__ == "__main__":
    print_title("DG + BDFν")

    convType = ConvType.SPATIAL

    # Data
    alpha = Constant(1.0)
    d_ext = Constant(1e-3) if convType==ConvType.TEMPORAL else Constant(1.0) 
    D     = d_ext*Identity(2)
    t0    = 0.0

    # Model parameters
    eta_0 = 5.0

    # Solver parameters
    tol   = 1e-12
    maxIt = 200

    # Exact solution
    c_space = lambda x: 0.25*(cos(2*pi*x[0])*cos(2*pi*x[1]) + 2.0)
    if convType == ConvType.TEMPORAL:
        c_ex = lambda x, t: c_space(x)*exp(-t)
    else:
        c_ex = lambda x, t: c_space(x)*(1.0 - t)

    if convType == ConvType.SPATIAL:
        print_subtitle("Space convergence test — linear time profile")

        # Space convergence parameters 
        N_ref    = [2, 3, 4]
        N_list   = [2**n for n in N_ref]
        l_space  = 4 
        T_space  = 1e-1
        dt_space = 1e-2
        nu_space = 1
        parameters["form_compiler"]["quadrature_degree"] = l_space**2 + 4

        # Storage variables
        errors_space_L2 = []
        errors_space_DG = []
        hs              = []
        N_el_list       = []

        # Loop over the meshes + benchmark
        with timer(f"Space convergence l={l_space}"):
            for N in N_list:
                print(f"\n --- N = {N} ---")
                mesh, _ = mesh_factory(mesh_type=MeshType.UNIT_SQUARE, N=N, structure=MeshStructure.STRUCTURED)
                N_el_list.append(mesh.num_cells())
                Solver = SolverDgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_ex, eta_0=eta_0)
                E_L2, E_DG, h = Solver.ConvergenceTest(
                    t0=t0, dt=dt_space, T=T_space, nu=nu_space, l=l_space, 
                    tol=tol, maxIt=maxIt
                )
                errors_space_L2.append(E_L2)
                errors_space_DG.append(E_DG)
                hs.append(h)

        # Print the rates
        print_space_rates(errors_space_L2, errors_space_DG, hs, N_el_list, l_space, method=Solver.SM)

        # Plot the rates
        plot_spatial_convergence(hs, errors_space_L2, errors_space_DG, l_space, method=Solver.SM, save=False)

    elif convType == ConvType.POLYNOMIAL:
        print_subtitle("Polynomial degree convergence test — linear time profile")

        # Polynomial degree convergence parameters 
        N_poly  = 8
        l_list  = [1, 2, 3]
        T_poly  = 2.5e-4
        dt_poly = 1e-5
        nu_poly = 4

        # Storage variable 
        errors_polynomial_L2 = []
        errors_polynomial_DG = []

        # Mesh
        mesh, _ = mesh_factory(mesh_type=MeshType.UNIT_SQUARE, N=N_poly, structure=MeshStructure.STRUCTURED)

        # Loop over l_list + benchmark
        with timer(f"Polynomial convergence"):
            for l in l_list:
                print(f"\n --- l = {l} ---")
                parameters["form_compiler"]["quadrature_degree"] = l**2 + 4
                Solver = SolverDgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_ex, eta_0=eta_0)
                E_L2, E_DG, h = Solver.ConvergenceTest(
                    t0=t0, dt=dt_poly, T=T_poly, nu=nu_poly, l=l, 
                    tol=tol, maxIt=maxIt
                )
                errors_polynomial_L2.append(E_L2)
                errors_polynomial_DG.append(E_DG)

        # Print polynomial fits
        print_polynomial_rates(errors_polynomial_L2, errors_polynomial_DG, l_list, method=Solver.SM)

        # Plot the rates for correct visualization 
        plot_polynomial_convergence(errors_polynomial_L2, errors_polynomial_DG, l_list, h, method=Solver.SM, save=False)

    elif convType == ConvType.TEMPORAL:
        print_subtitle("Time convergence test — exponential time profile")

        # Time convergence parameters 
        N_time  = 32
        l_time  = 4
        T_time  = 2
        dt_list = [0.5, 0.25, 0.125]
        nu_time = 6
        parameters["form_compiler"]["quadrature_degree"] = l_time**2 + 4

        # Storage variable 
        errors_time_L2 = []
        errors_time_DG = []

        # Mesh
        mesh, _ = mesh_factory(mesh_type=MeshType.UNIT_SQUARE, N=N_time, structure=MeshStructure.STRUCTURED)

        # Loop over dt_list + benchmark
        with timer(f"Time convergence ν={nu_time}"):
            for dt in dt_list:
                print(f"\n --- τ = {dt:.{get_decimals(dt)}f} ---")
                Solver = SolverDgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_ex, eta_0=eta_0)
                E_L2, E_DG, h = Solver.ConvergenceTest(
                    t0=t0, dt=dt, T=T_time, nu=nu_time, l=l_time, 
                    tol=tol, maxIt=maxIt
                )
                errors_time_L2.append(E_L2)
                errors_time_DG.append(E_DG)

        # Print the time convergence rates
        print_time_rates(errors_time_L2, errors_time_DG, dt_list, nu_time, time_method=Solver.TM, space_method=Solver.SM)

        # Plot the rates
        plot_time_convergence(dt_list, errors_time_L2, errors_time_DG, nu_time, method=Solver.TM, space_method=Solver.SM, save=False)
