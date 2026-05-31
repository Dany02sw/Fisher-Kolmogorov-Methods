"""
Forward simulation of α-synuclein spreading on a 2-D brain section.

Edit the block delimited by CONFIGURATION / END CONFIGURATION and run:

    python main_brain.py
"""

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

from models.solver_ldg_theta import SolverLdgTheta   # swap to any solver with a Solve method

from utilities.enum_utilities import PolyDegree, BdfOrder, ThetaMethod

SOLVER_CLASS = SolverLdgTheta

# Time integration
T   = 50.0
dt  = 2.5e-1
tht = 0.5
t0  = 0.0

# FE polynomial degree
l = PolyDegree

# Model-specific parameters (LDG)
C11 = 10.0
C12 = 0.5

# Solver
tol   = 1e-6
max_it = 500

# ── END CONFIGURATION ─────────────────────────────────────────────────────────

from dolfin import *

from meshes.mesh_import            import mesh_factory
from utilities.enum_utilities      import BrainSection, MeshType
from utilities.initial_conditions  import get_initial_condition
from utilities.profiling_utilities import timer
from utilities.print_utilities     import print_title, print_subtitle


def build_brain_pde_data(plane: BrainSection):
    """
    Assemble the spatially varying PDE coefficients (alpha, D) for a given
    brain section, using the subdomain tags embedded in the mesh.

    Parameters
    ----------
    plane : BrainSection
        Brain cross-section to simulate on.

    Returns
    -------
    mesh        : dolfin Mesh
    subdomains  : dolfin MeshFunction
    alpha       : UFL expression for the reaction coefficient
    D           : UFL expression for the diffusion tensor
    c_0         : initial condition (dolfin Function or Expression)
    """
    mesh, subdomains = mesh_factory(mesh_type=MeshType.BRAIN_2D, brain_plane=plane)
    c_0              = get_initial_condition(plane=plane)

    DG0                         = FunctionSpace(mesh, "DG", 0)
    subdomains_tags             = Function(DG0)
    subdomains_tags.vector()[:] = subdomains.array()

    alpha_grey  = Constant(0.5)
    alpha_white = Constant(1.0)
    alpha       = conditional(le(subdomains_tags, 1.5), alpha_grey, alpha_white)

    a       = as_tensor((1.0, 1.0))
    d_ext   = Constant(8e-3)
    d_axn   = Constant(8e-2)
    D_grey  = d_ext * Identity(2)
    D_white = d_ext * Identity(2) + d_axn * outer(a, a)
    D       = conditional(le(subdomains_tags, 1.5), D_grey, D_white)

    return mesh, subdomains, alpha, D, c_0


if __name__ == "__main__":
    print_title("α-synuclein spreading — brain simulation")

    plane = BrainSection.SAGITTAL
    mesh, _, alpha, D, c_0 = build_brain_pde_data(plane)

    parameters["form_compiler"]["quadrature_degree"] = l ** 2 + 4

    print_subtitle(f"Spreading on {mesh.name()} section  ·  {SOLVER_CLASS.__name__}")

    solver = SOLVER_CLASS(mesh, D, alpha, c_0, C11=C11, C12=C12)
    with timer(f"Spreading on {mesh.name()} section"):
        solver.Solve(t0=t0, dt=dt, T=T, tht=tht, l=l, tol=tol, maxIt=max_it)
