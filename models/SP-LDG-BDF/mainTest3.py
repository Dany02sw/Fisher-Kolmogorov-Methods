from models.solver_spldg_bdf import SolverSpLdgBDF

from dolfin import *

from meshes.mesh_import            import mesh_factory
from utilities.enum_utilities      import BrainSection, MeshType
from utilities.initial_conditions  import get_initial_condition
from utilities.profiling_utilities import timer
from utilities.print_utilities     import print_title, print_subtitle

if __name__ == "__main__":
    print_title("SP-LDG + BDFν")

    # Mesh import
    plane            = BrainSection.SAGITTAL  # Plane section to simulate on
    mesh, subdomains = mesh_factory(mesh_type=MeshType.BRAIN_2D, brain_plane=plane)
    c_0              = get_initial_condition(plane=plane)

    # DG0 function to store subdomain tags
    DG0                         = FunctionSpace(mesh, 'DG', 0)
    subdomains_tags             = Function(DG0)
    subdomains_tags.vector()[:] = subdomains.array() 

    # Reaction coefficient
    alpha_grey  = Constant(0.5)   # Reaction coefficient in the grey matter [alpha_grey_matter] = 1/year 
    alpha_white = Constant(1.0)   # Reaction coefficient in the grey matter [alpha_grey_matter] = 1/year
    alpha       = conditional(
        le(subdomains_tags, 1.5), alpha_grey, alpha_white
    )

    # Diffusion tensor
    a       = as_tensor((1.0, 1.0))  # Axonal directions 
    d_ext   = Constant(8e-3)         # [d_ext] = m^{2}/year
    d_axn   = Constant(8e-2)         # [d_axn] = m^{2}/year
    D_grey  = d_ext*Identity(2)                     # Tensor in the grey matter (Identity is already an as_tensor)
    D_white = d_ext*Identity(2) + d_axn*outer(a, a) # Tensor in the grey matter (Identity is already an as_tensor, outer dives us the cross product)
    D       = conditional(
        le(subdomains_tags, 1.5), D_grey, D_white
    )
    
    # Other data
    l   = 2
    t0  = 0.0
    T   = 50.0
    dt  = 2.5e-1
    nu  = 6

    # Model parameters
    eps       = 0.0
    eta_0     = 2.0 
    theta     = 0.5 
    smoothing = 1e-9

    # Solver parameters
    tol = 1e-6
    maxIt = 500

    # Set the quadrature degree
    parameters["form_compiler"]["quadrature_degree"] = l**2 + 4

    # Solve the problem 
    print_subtitle(f"Spreading of α-synuclein on {mesh.name()} section")
    Solver = SolverSpLdgBDF(mesh=mesh, D=D, alpha=alpha, c_0=c_0, eps=eps, eta_0=eta_0, theta=theta, smoothing=smoothing)
    with timer(f"Spreading of α-synuclein on {mesh.name()} section"):
        Solver.Solve(t0=t0, dt=dt, T=T, nu=nu, l=l, tol=tol, maxIt=maxIt)