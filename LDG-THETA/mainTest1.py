from SolverLdgTheta import SolverLdgTheta
from dolfin import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Meshes import create_unite_square_mesh
from EnumUtilities import ConvType
from PlotUtilities import plot_polynomial_convergence
from FunctionUtilities import compute_rate, compute_exponential_fit

if __name__ == "__main__":
    print("\n")
    print("#"*59)
    print(23*"#"+" LDG + THETA "+ 23*"#")
    print("#"*59)

    convType = ConvType.TEMPORAL

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
        N_ref     = [2, 3, 4]
        N_list    = [2**n for n in N_ref]
        l_space   = 1 
        T_space   = 3e-2
        dt_space  = 1e-3
        tht_space = 0.5
        parameters["form_compiler"]["quadrature_degree"] = l_space**2 + 4

        # Storage variables
        errors_space_c = []
        errors_space_q = []
        hs = []

        # Loop over the meshes 
        for N in N_list:
            print(f"\n --- N = {N} ---")
            mesh = create_unite_square_mesh(N, unstructured=False, plots=False)
            SLT  = SolverLdgTheta(mesh, D, alpha, C11, C12, c_ex)
            E_c, E_q, h = SLT.ConvergenceTest(
                t0=t0, dt=dt_space, T=T_space, tht=tht_space, l=l_space, 
                tol=tol, maxIt=maxIt
            )
            errors_space_c.append(E_c)
            errors_space_q.append(E_q)
            hs.append(h)

        # Loop to print space rates for E_c 
        print(f"\n=== Space convergence rates for l = {l_space} (t = T) ===")
        print("\n--- Rates for E_c ---")
        for i in range(1, len(N_list)):
            rate_c = compute_rate(errors_space_c, hs, i)
            print(f"N={N_list[i-1]} → N={N_list[i]}: E_c rate ≈ {rate_c:.2f} (expected = {l_space+1:.2f})")

        # Loop to print space rates for D*grad(c) 
        print("\n--- Rates for E_q ---")
        for i in range(1, len(N_list)):
            rate_q = compute_rate(errors_space_q, hs, i)
            print(f"N={N_list[i-1]} → N={N_list[i]}: E_q rate ≈ {rate_q:.2f} (expected = {l_space:.2f})")

    elif convType == ConvType.POLYNOMIAL:
        # ======================================================================
        # === 2) CONVERGENCE WRT THE POLYNOMIAL DEGREE (linear time profile) ===
        # ======================================================================
        print("\n" + "="*69)
        print("===== Polynomial degree convergence test (time profile: linear) =====")
        print("="*69)

        # Polynomial degree convergence parameters 
        N_poly   = 8
        l_list   = [1, 2, 3]
        T_poly   = 2.5e-4
        dt_poly  = 1e-5
        tht_poly = 0.5

        # Storage variable 
        errors_polynomial_c = []
        errors_polynomial_q = []

        # Mesh
        mesh = create_unite_square_mesh(N_poly, unstructured=False, plots=False)

        # Loop over l_list 
        for l in l_list:
            print(f"\n --- l = {l} ---")
            parameters["form_compiler"]["quadrature_degree"] = l**2 + 4
            SLT  = SolverLdgTheta(mesh, D, alpha, C11, C12, c_ex)
            E_c, E_q, h = SLT.ConvergenceTest(
                t0=t0, dt=dt_poly, T=T_poly, tht=tht_poly, l=l, 
                tol=tol, maxIt=maxIt
            )
            errors_polynomial_c.append(E_c)
            errors_polynomial_q.append(E_q)

        # Loop to print convergence rates wrt polynomial degree for c 
        print("\n=== Polynomial degree convergence rates (t = T) ===")
        beta_c, fitted_c, res_c = compute_exponential_fit(errors_polynomial_c, l_list)
        print("\n--- Rates for E_c ---")
        for i in range(len(l_list)):
            print(f"  l={l_list[i]}: E_c = {errors_polynomial_c[i]:.4e}  (fit = {fitted_c[i]:.4e})")
        print(f"  → E_c ~ exp(-{beta_c:.2f} * l),  residual = {res_c:.2e}")

        # Loop to print convergence rates wrt polynomial degree for D*grad(c) 
        beta_q, fitted_q, res_q = compute_exponential_fit(errors_polynomial_q, l_list)
        print("\n--- Rates for E_q ---")
        for i in range(len(l_list)):
            print(f"  l={l_list[i]}: E_q = {errors_polynomial_q[i]:.4e}  (fit = {fitted_q[i]:.4e})")
        print(f"  → E_q ~ exp(-{beta_q:.2f} * l),  residual = {res_q:.2e}")

        # Plot the rates for correct visualization 
        plot_polynomial_convergence(errors_polynomial_c, errors_polynomial_q, l_list, h, save=False)

    elif convType == ConvType.TEMPORAL:
        # ======================================================
        # === 3) TIME CONVERGENCE (exponential time profile) ===
        # ======================================================
        print("\n" + "="*61)
        print("===== Time convergence test (time profile: exponential) =====")
        print("="*61)

        # Time convergence parameters 
        N_time   = 32
        l_time   = 2
        T_time   = 2
        dt_list  = [0.5, 0.25, 0.125]
        tht_time = 0.5
        parameters["form_compiler"]["quadrature_degree"] = l_time**2 + 4

        # Storage variable 
        errors_time_c = []
        errors_time_q = []

        # Mesh
        mesh = create_unite_square_mesh(N_time, unstructured=False, plots=False)

        # Loop over dt_list 
        print(f"\n>>> Running theta = {tht_time} ...")
        for dt in dt_list:
            print(f"\n --- dt = {dt:.4f} ---")
            SLT  = SolverLdgTheta(mesh, D, alpha, C11, C12, c_ex)
            E_c, E_q, h = SLT.ConvergenceTest(
                t0=t0, dt=dt, T=T_time, tht=tht_time, l=l_time, 
                tol=tol, maxIt=maxIt
            )
            errors_time_c.append(E_c)
            errors_time_q.append(E_q)

        # Loop to print time convergence rates for c 
        print(f"\n=== Time convergence rates for theta = {tht_time} (t = T) ===")
        expected_time_rate = 2.0 if tht_time==0.5 else 1.0
        print("\n--- Rates for E_c ---")
        for i in range(1, len(dt_list)):
            rate_c = compute_rate(errors_time_c, dt_list, i)
            print(f"dt={dt_list[i-1]} → {dt_list[i]}: E_c rate ≈ {rate_c:.3f} (expected {expected_time_rate:.2f})")

        # Loop to print time convergence rates for D*grad(c) 
        print("\n--- Rates for E_q ---")
        for i in range(1, len(dt_list)):
            rate_q = compute_rate(errors_time_q, dt_list, i)
            print(f"dt={dt_list[i-1]} → {dt_list[i]}: E_q rate ≈ {rate_q:.3f} (expected {expected_time_rate:.2f})")
