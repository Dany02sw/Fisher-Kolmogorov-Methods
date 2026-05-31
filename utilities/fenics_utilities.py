from dolfin import *

# Local project
def LocalProject(v, V):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u

# Normalization
def Normalize(c_):
    c_.vector()[:] -= c_.vector().min()
    c_.vector()[:] /= (c_.vector().max() - c_.vector().min())
    return c_

# Weighted average operator
def wavg(gamma, v_):
    return (1.0-gamma)*v_('+') + gamma*v_('-')

# Harmonic average operator
def havg(v_):
    return (2.0*v_('+')*v_('-'))/(v_('+') + v_('-'))

# Function to compute a smoothed max
def smoothMax(a, b, epsilon=Constant(1e-10)):
    return (a + b + sqrt((a - b)*(a - b) + epsilon)) / 2.0