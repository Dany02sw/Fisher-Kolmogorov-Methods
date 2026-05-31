from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Union
from dolfin      import *

from fisher_kolmogorov.utilities.enum_utilities import PolyDegree, BdfOrder, ThetaMethod, MeshType, MeshStructure, BrainSection

# Class for storing hyperparameters ________________________________________________________________________________________________________
@dataclass
class ConvergenceParams:
    """
    Hyperparameters for a single convergence study (spatial, polynomial, or temporal).

    Attributes:
        N_ref         : exponents for mesh refinements, i.e. N = 2**n  (spatial)
        l_space       : polynomial degree used in the spatial study
        T             : final simulation time
        dt            : time step size (spatial / polynomial studies)
        nu_or_tht     : BDF order or theta value depending on the time integrator
        dt_list       : list of time steps for the temporal study
        l_list        : list of polynomial degrees for the polynomial study
        N_fixed       : fixed mesh size N used when the mesh is not refined (polynomial / temporal)
        mesh_type     : mesh geometry
        mesh_structure: structured vs unstructured
    """
    N_ref         : list       = field(default_factory=lambda: [2, 3, 4])
    l_space       : PolyDegree = PolyDegree.P3
    T             : float      = 1e-1
    dt            : float      = 1e-2
    nu_or_tht     : Union[BdfOrder, ThetaMethod] = BdfOrder.BDF1
    dt_list       : list                         = field(default_factory=lambda: [0.5, 0.25, 0.125])
    l_list        : list                         = field(default_factory=lambda: [PolyDegree.P1, PolyDegree.P2, PolyDegree.P3])
    N_fixed       : int           = 8
    mesh_type     : MeshType      = MeshType.UNIT_SQUARE
    mesh_structure: MeshStructure = MeshStructure.UNSTRUCTURED


# Class for the total physical configuration of a convergence test ________________________________________________________________________
@dataclass
class TestConfig:
    """
    Full specification of a convergence test: physical data, exact solution,
    and per-study hyperparameters.

    Attributes:
        name          : human-readable label used in print headers
        alpha         : reaction coefficient (dolfin Constant or expression)
        D_factory     : callable (d_ext, mesh) -> diffusion tensor; receives a
                        dolfin Constant for d_ext and may use mesh metadata
        c_exact       : callable (x, t) -> exact solution UFL expression
        t0            : initial time
        d_ext         : diffusivity used in convergence tests
        spatial       : hyperparameters for the spatial convergence study
        polynomial    : hyperparameters for the polynomial convergence study
        temporal      : hyperparameters for the temporal convergence study
        mesh_kwargs   : extra keyword arguments forwarded to mesh_factory
                        (e.g. P1, P2 for rectangular meshes)
    """
    name       : str
    alpha      : object
    D_factory  : callable
    c_exact    : callable
    t0         : float = 0.0
    d_ext      : float = 1.0
    spatial    : ConvergenceParams = field(default_factory=ConvergenceParams)
    polynomial : ConvergenceParams = field(default_factory=ConvergenceParams)
    temporal   : ConvergenceParams = field(default_factory=ConvergenceParams)
    mesh_kwargs: dict = field(default_factory=dict)


# Class for storing hyperparameters of a simulation _________________________________________________________________________________________
@dataclass
class SimulationParams:
    """
    Numerical parameters for a forward simulation run.

    Attributes:
        l         : FE polynomial degree
        T         : final simulation time
        dt        : time step size
        nu_or_tht : time integrator parameter (BDF order or theta value)
        t0        : initial time
        tol       : nonlinear solver tolerance
        max_it    : maximum nonlinear solver iterations
    """
    l         : PolyDegree                   = PolyDegree.P2
    T         : float                        = 50.0
    dt        : float                        = 2.5e-1
    nu_or_tht : Union[BdfOrder, ThetaMethod] = ThetaMethod.CN
    t0        : float                        = 0.0
    tol       : float                        = 1e-6
    max_it    : int                          = 500


# Class for a configuration of a brain simulation __________________________________________________________________________________________
@dataclass
class BrainConfig:
    alpha_grey  : float = 0.5
    alpha_white : float = 1.0
    d_ext       : float = 8e-3
    d_axn       : float = 8e-2
    a           : tuple = (1.0, 1.0)
    plane       : BrainSection = BrainSection.SAGITTAL
