from SolverLdgTheta import SolverLdgTheta
from Meshes import create_rectangle_mesh
from dolfin import *
import numpy as np
from ufl import tanh
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plotClass import ConvType
from plotUtilities import plot_polynomial_convergence

if __name__ == "__main__":
    print("\n")
    print("#"*59)
    print(23*"#"+" LDG + THETA "+ 23*"#")
    print("#"*59)

    convType = ConvType.SPATIAL

    # Data
    alpha = Constant(1.0)
    d_ext = Constant(1e-3) 
    D     = d_ext*Identity(2)
    v     = Constant(5.0*sqrt(alpha*d_ext/6.0))
    C11   = 10.0
    C12   = 0.5
    t0    = 0.0

    # Solver parameters
    tol = 1e-8
    maxIt = 300

    # Exact solution
    c_ex = lambda x, t: 0.25*(1.0 + tanh(8.0 - sqrt(alpha/(24.0*d_ext))*(x[0] - v*t)))**2

    if convType == ConvType.SPATIAL:
        # ==================================================
        # === 1) SPACE CONVERGENCE (linear time profile) ===
        # ==================================================
        print("\n" + "="*36)
        print("====== Space convergence test ======")
        print("="*36)

        # Space convergence parameters 
        N_ref     = [3, 4, 5]
        N_list    = [2**n for n in N_ref]
        l_space   = 2 
        T_space   = 3e-3
        dt_space  = 1e-4
        tht_space = 0.5
        parameters["form_compiler"]["quadrature_degree"] = l_space**2 + 4

        # Storage variables
        errors_space_c = []
        errors_space_q = []
        hs = []

        # Loop over the meshes 
        for N in N_list:
            print(f"\n --- N = {N} ---")
            mesh = create_rectangle_mesh(N, unstructured=False, plots=False)
            SLT  = SolverLdgTheta(mesh, D, alpha, C11, C12, c_ex)
            E_c, E_q, h = SLT.ConvergenceTest(
                t0=t0, dt=dt_space, T=T_space, l=l_space, 
                tol=tol, maxIt=maxIt
            )
            errors_space_c.append(E_c)
            errors_space_q.append(E_q)
            hs.append(h)

        # Loop to print space rates for E_c 
        print(f"\n=== Space convergence rates for l = {l_space} (t = T) ===")
        print("\n Rates for E_c ")
        for i in range(1, len(N_list)):
            rate_c = np.log(errors_space_c[i-1]/errors_space_c[i]) / np.log(hs[i-1]/hs[i])
            print(f"N={N_list[i-1]} → N={N_list[i]}: E_c rate ≈ {rate_c:.2f} (expected = {l_space+1:.2f})")

        # Loop to print space rates for D*grad(c) 
        print("\n Rates for E_q ")
        for i in range(1, len(N_list)):
            rate_q = np.log(errors_space_q[i-1]/errors_space_q[i]) / np.log(hs[i-1]/hs[i])
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
        T_poly   = 10.0
        dt_poly  = 2.5e-2
        tht_poly = 0.5

        # Storage variable 
        errors_polynomial_c = []
        errors_polynomial_q = []

        # Mesh
        mesh = create_rectangle_mesh(N_poly, unstructured=False, plots=False)

        # Loop over l_list 
        for l in l_list:
            print(f"\n --- l = {l} ---")
            parameters["form_compiler"]["quadrature_degree"] = l**2 + 4
            SLT  = SolverLdgTheta(mesh, D, alpha, C11, C12, c_ex)
            E_c, E_q, h = SLT.ConvergenceTest(
                t0=t0, dt=dt_poly, T=T_poly, l=l, 
                tol=tol, maxIt=maxIt
            )
            errors_polynomial_c.append(E_c)
            errors_polynomial_q.append(E_q)

        # Loop to print convergence rates wrt polynomial degree for c 
        print("\n=== Polynomial degree convergence rates (t = T) ===")
        print("\n Rates for E_c ")
        for i in range(1, len(l_list)):
            rate_c = np.log(errors_polynomial_c[i-1]/errors_polynomial_c[i]) / np.log(l_list[i]/l_list[i-1])
            print(f"l={l_list[i-1]} → l={l_list[i]}: E_c rate ≈ {rate_c:.2f} (expected = exponential decay)")

        # Loop to print convergence rates wrt polynomial degree for D*grad(c) 
        print("\n Rates for E_q ")
        for i in range(1, len(l_list)):
            rate_q = np.log(errors_polynomial_q[i-1]/errors_polynomial_q[i]) / np.log(l_list[i]/l_list[i-1])
            print(f"l={l_list[i-1]} → l={l_list[i]}: E_q rate ≈ {rate_q:.2f} (expected = exponential decay)")

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
        T_time   = 3
        dt_list  = [0.5, 0.25, 0.125]
        tht_time = 0.5
        parameters["form_compiler"]["quadrature_degree"] = l_time**2 + 4

        # Storage variable 
        errors_time_c = []
        errors_time_q = []

        # Mesh
        mesh = create_rectangle_mesh(N_time, unstructured=False, plots=False)

        # Loop over dt_list 
        print(f"\n>>> Running theta = {tht_time} ...")
        for dt in dt_list:
            print(f"\n --- dt = {dt:.4f} ---")
            SLT  = SolverLdgTheta(mesh, D, alpha, C11, C12, c_ex)
            E_c, E_q, h = SLT.ConvergenceTest(
                t0=t0, dt=dt, T=T_time, l=l_time, 
                tol=tol, maxIt=maxIt
            )
            errors_time_c.append(E_c)
            errors_time_q.append(E_q)

        # Loop to print time convergence rates for c 
        print(f"\n=== Time convergence rates for theta = {tht_time} (t = T) ===")
        print("\n Rates for E_c ")
        for i in range(1, len(dt_list)):
            rate = np.log(errors_time_c[i-1] / errors_time_c[i]) / np.log(dt_list[i-1] / dt_list[i])
            print(f"dt={dt_list[i-1]} → {dt_list[i]}: E_c rate ≈ {rate:.3f} (expected {1.0/tht_time:.2f})")

        # Loop to print time convergence rates for D*grad(c) 
        print("\n Rates for E_q ")
        for i in range(1, len(dt_list)):
            rate = np.log(errors_time_q[i-1] / errors_time_q[i]) / np.log(dt_list[i-1] / dt_list[i])
            print(f"dt={dt_list[i-1]} → {dt_list[i]}: E_sigma rate ≈ {rate:.3f} (expected {1.0/tht_time:.2f})")