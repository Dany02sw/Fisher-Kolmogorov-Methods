from dolfin import *

from utilities.fenics_utilities import wavg

# Build the form associated to the implicit definition of grad_LDG
def grad_LDG(v, phi, n, gamma, dx, dS):
    return inner(grad(v), phi)*dx - inner(jump(v, n), wavg(1.0-gamma, phi))*dS

# Build the bilinear form LDG
def inner_LDG(w, psi, n, gamma, h_avg, D, dx, dS, alpha, sym=True):

    # First term
    mass_term = alpha*inner(w, psi)*dx

    # Second term
    if sym:
        # Symmetric
        gradpsi_w = grad_LDG(psi, D*grad(w),  n, gamma, dx, dS)
        gradw_psi = grad_LDG(w, dot(D, grad(psi)), n, gamma, dx, dS)
        gradgrad_term = 0.5*(gradw_psi + gradpsi_w)

    else:
        # Non symmetric
        gradgrad_term = grad_LDG(w, dot(D, grad(psi)), n, gamma, dx, dS)

    # Third term
    j_h = inner((1.0/h_avg)*jump(w,n), jump(psi,n))*dS 

    return mass_term + gradgrad_term + j_h

# Build the form associated to the implicit definition of div_LDG
def div_LDG(r, psi, n, gamma, dx, dS):
    return - inner(r, grad(psi))*dx + inner(wavg(1-gamma, r), jump(psi, n))*dS