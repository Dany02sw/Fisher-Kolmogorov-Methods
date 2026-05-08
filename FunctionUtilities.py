import numpy as np
from dolfin import *

# Function to compute rates
def compute_rate(errors, refinements, i):
    return np.log(errors[i-1] / errors[i]) / np.log(refinements[i-1] / refinements[i])

# Function to compute and print exponential fit wrt polynomial degree
def compute_exponential_fit(errors, l_list):
    ls       = np.array(l_list, dtype=float)
    log_e    = np.log(np.array(errors))
    coeffs   = np.polyfit(ls, log_e, deg=1)
    beta     = -coeffs[0]
    fitted   = np.exp(np.polyval(coeffs, ls))
    residual = np.max(np.abs(log_e - np.polyval(coeffs, ls)))
    return beta, fitted, residual

# Local project
def LocalProject(v, V, solver=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    if solver is None:
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u, solver

# Normalization
def Normalize(c_):
    c_.vector()[:] -= c_.vector().min()
    c_.vector()[:] /= (c_.vector().max() - c_.vector().min())
    return c_