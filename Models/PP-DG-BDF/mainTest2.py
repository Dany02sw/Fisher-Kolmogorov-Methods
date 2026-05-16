from SolverPpDgBDF import SolverPpDgBDF

from dolfin import *
from ufl import tanh

from Meshes.Meshes import create_rectangle_mesh
from Utilities.ProfilingUtilities import timer
from Utilities.EnumUtilities import ConvType
from Utilities.PlotUtilities import plot_spatial_convergence, plot_polynomial_convergence, plot_time_convergence
from Utilities.PrintUtilities import print_space_rates, print_polynomial_rates, print_time_rates, print_title, print_subtitle
from Utilities.MathUtilities import get_decimals

if __name__ == "__main__":
    print_title("PP-DG + BDFν")

    convType = ConvType.POLYNOMIAL

    # Data
    alpha = Constant(1.0)
    d_ext = Constant(1e-3) 
    D     = d_ext*Identity(2)
    v     = Constant(5.0*sqrt(alpha*d_ext/6.0))
    t0    = 0.0

    # Model parameters
    eps       = 0.0
    eta_0     = 1.0
    smoothing = 0.0

    # Solver parameters
    tol   = 1e-10
    maxIt = 300

    # Exact solution
    c_ex = lambda x, t: 0.25*(1.0 + tanh(8.0 - sqrt(alpha/(24.0*d_ext))*(x[0] - v*t)))**2

    # Mesh verteces
    P1 = Point((0.0, 0.0))
    P2 = Point((3.0, 1.0))

    if convType == ConvType.SPATIAL:
        print_subtitle("Space convergence test - waves")

        # Space convergence parameters 
        N_ref    = [3, 4, 5]
        N_list   = [2**n for n in N_ref]
        l_space  = 2 
        T_space  = 3e-1
        dt_space = 1e-2
        nu_space = 3
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
                mesh = create_rectangle_mesh(N, P1, P2, unstructured=False, plotMesh=False)
                N_el_list.append(mesh.num_cells())
                Solver  = SolverPpDgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_ex, eps=eps, eta_0=eta_0, smoothing=smoothing)
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
        print_subtitle("Polynomial degree convergence test — waves")

        # Polynomial degree convergence parameters 
        N_poly  = 8
        l_list  = [1, 2]
        T_poly  = 10.0
        dt_poly = 2.5e-2
        nu_poly = 4

        # Storage variable 
        errors_polynomial_L2 = []
        errors_polynomial_DG = []

        # Mesh
        mesh = create_rectangle_mesh(N_poly, P1, P2, unstructured=False, plotMesh=False)

        # Loop over l_list + benchmark
        with timer(f"Polynomial convergence"):
            for l in l_list:
                print(f"\n --- l = {l} ---")
                parameters["form_compiler"]["quadrature_degree"] = l**2 + 4
                Solver = SolverPpDgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_ex, eps=eps, eta_0=eta_0, smoothing=smoothing)
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
        l_time  = 2
        T_time  = 2
        dt_list = [0.5, 0.25, 0.125]
        nu_time = 6
        parameters["form_compiler"]["quadrature_degree"] = l_time**2 + 4

        # Storage variable 
        errors_time_L2 = []
        errors_time_DG = []

        # Mesh
        mesh = create_rectangle_mesh(N_time, P1, P2, unstructured=False, plotMesh=False)

        # Loop over dt_list + benchmark
        with timer(f"Time convergence ν={nu_time}"):
            for dt in dt_list:
                print(f"\n --- τ = {dt:.{get_decimals(dt)}f} ---")
                Solver = SolverPpDgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_ex, eps=eps, eta_0=eta_0, smoothing=smoothing)
                E_L2, E_DG, h = Solver.ConvergenceTest(
                    t0=t0, dt=dt, T=T_time, nu=nu_time, l=l_time, 
                    tol=tol, maxIt=maxIt
                )
                errors_time_L2.append(E_L2)
                errors_time_DG.append(E_DG)

        # Print the time convergence rates
        print_time_rates(errors_time_L2, errors_time_DG, dt_list, nu_time, time_method=Solver.TM)

        # Plot the rates
        plot_time_convergence(dt_list, errors_time_L2, errors_time_DG, nu_time, method=Solver.TM, space_method=Solver.SM, save=False)
