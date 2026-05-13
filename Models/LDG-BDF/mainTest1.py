from SolverLdgBDF import SolverLdgBDF

from dolfin import *

from Meshes.Meshes import create_unite_square_mesh
from Utilities.ProfilingUtilities import timer
from Utilities.EnumUtilities import ConvType
from Utilities.PlotUtilities import plot_spatial_convergence, plot_polynomial_convergence, plot_time_convergence
from Utilities.PrintUtilities import print_space_rates, print_polynomial_rates, print_time_rates

if __name__ == "__main__":
    print("\n")
    print("#"*57)
    print(23*"#"+" LDG + BDF "+ 23*"#")
    print("#"*57)

    convType = ConvType.SPATIAL

    # Data
    alpha = Constant(1.0)
    d_ext = Constant(1e-3) if convType==ConvType.TEMPORAL else Constant(1.0) 
    D     = d_ext*Identity(2)
    C11   = 10.0
    C12   = 0.5
    t0    = 0.0

    # Solver parameters
    tol = 1e-12
    maxIt = 200

    # Exact solution
    c_space = lambda x: 0.25*(cos(2*pi*x[0])*cos(2*pi*x[1]) + 2.0)
    if convType == ConvType.TEMPORAL:
        c_ex = lambda x, t: c_space(x)*exp(-t)
    else:
        c_ex = lambda x, t: c_space(x)*(1.0 - t)

    if convType == ConvType.SPATIAL:
        # ==================================================
        # === 1) SPACE CONVERGENCE (linear time profile) ===
        # ==================================================
        print("\n" + "="*59)
        print("====== Space convergence test (time profile: linear) ======")
        print("="*59)

        # Space convergence parameters 
        N_ref    = [2, 3, 4]
        N_list   = [2**n for n in N_ref]
        l_space  = 1 
        T_space  = 3e-2
        dt_space = 1e-3
        nu_space = 4
        parameters["form_compiler"]["quadrature_degree"] = l_space**2 + 4

        # Storage variables
        errors_space_c = []
        errors_space_q = []
        hs             = []
        N_el_list      = []

        # Loop over the meshes + benchmark
        with timer(f"Space convergence l={l_space}"):
            for N in N_list:
                print(f"\n --- N = {N} ---")
                mesh = create_unite_square_mesh(N, unstructured=False, plotMesh=False)
                N_el_list.append(mesh.num_cells())
                SLT  = SolverLdgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_ex, C11=C11, C12=C12)
                E_c, E_q, h = SLT.ConvergenceTest(
                    t0=t0, dt=dt_space, T=T_space, nu=nu_space, l=l_space, 
                    tol=tol, maxIt=maxIt
                )
                errors_space_c.append(E_c)
                errors_space_q.append(E_q)
                hs.append(h)

        # Print the rates
        print_space_rates(errors_space_c, errors_space_q, hs, N_el_list, l_space, method=SLT.SM)

        # Plot the rates
        plot_spatial_convergence(hs, errors_space_c, errors_space_q, l_space, method=SLT.SM, save=False)

    elif convType == ConvType.POLYNOMIAL:
        # ======================================================================
        # === 2) CONVERGENCE WRT THE POLYNOMIAL DEGREE (linear time profile) ===
        # ======================================================================
        print("\n" + "="*69)
        print("===== Polynomial degree convergence test (time profile: linear) =====")
        print("="*69)

        # Polynomial degree convergence parameters 
        N_poly  = 8
        l_list  = [1, 2, 3]
        T_poly  = 2.5e-4
        dt_poly = 1e-5
        nu_poly = 0.5

        # Storage variable 
        errors_polynomial_c = []
        errors_polynomial_q = []

        # Mesh
        mesh = create_unite_square_mesh(N_poly, unstructured=False, plotMesh=False)

        # Loop over l_list + benchmark
        with timer(f"Polynomial convergence"):
            for l in l_list:
                print(f"\n --- l = {l} ---")
                parameters["form_compiler"]["quadrature_degree"] = l**2 + 4
                SLT  = SolverLdgBDF(mesh, D, alpha, C11, C12, c_ex)
                E_c, E_q, h = SLT.ConvergenceTest(
                    t0=t0, dt=dt_poly, T=T_poly, nu=nu_poly, l=l, 
                    tol=tol, maxIt=maxIt
                )
                errors_polynomial_c.append(E_c)
                errors_polynomial_q.append(E_q)

        # Print polynomial fits
        print_polynomial_rates(errors_polynomial_c, errors_polynomial_q, l_list, method=SLT.SM)

        # Plot the rates for correct visualization 
        plot_polynomial_convergence(errors_polynomial_c, errors_polynomial_q, l_list, h, method=SLT.SM, save=False)

    elif convType == ConvType.TEMPORAL:
        # ======================================================
        # === 3) TIME CONVERGENCE (exponential time profile) ===
        # ======================================================
        print("\n" + "="*61)
        print("===== Time convergence test (time profile: exponential) =====")
        print("="*61)

        # Time convergence parameters 
        N_time  = 32
        l_time  = 2
        T_time  = 2
        dt_list = [0.5, 0.25, 0.125]
        nu_time = 1.0
        parameters["form_compiler"]["quadrature_degree"] = l_time**2 + 4

        # Storage variable 
        errors_time_c = []
        errors_time_q = []

        # Mesh
        mesh = create_unite_square_mesh(N_time, unstructured=False, plotMesh=False)

        # Loop over dt_list + benchmark
        with timer(f"Time convergence ν={nu_time}"):
            for dt in dt_list:
                print(f"\n --- dt = {dt:.4f} ---")
                SLT  = SolverLdgBDF(mesh, D, alpha, C11, C12, c_ex)
                E_c, E_q, h = SLT.ConvergenceTest(
                    t0=t0, dt=dt, T=T_time, nu=nu_time, l=l_time, 
                    tol=tol, maxIt=maxIt
                )
                errors_time_c.append(E_c)
                errors_time_q.append(E_q)

        # Print the time convergence rates
        print_time_rates(errors_time_c, errors_time_q, dt_list, nu_time, time_method=SLT.TM)

        # Plot the rates
        plot_time_convergence(dt_list, errors_time_c, errors_time_q, nu_time, method=SLT.TM, space_method=SLT.SM, save=False)
