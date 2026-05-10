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