from fisher_kolmogorov.models.solver_factory import make_solver_class
from fisher_kolmogorov.models.solver_base    import SolverBase

from fisher_kolmogorov.utilities.enum_utilities   import SpaceMethod, TimeMethod
from fisher_kolmogorov.configs.model_configs       import DgParams, LdgParams, PpDgParams, SpLdgParams

# Pre-built solver classes for direct import, mirroring the old concrete classes.
# These are the canonical names to use throughout the codebase.
SolverDgBdf        = make_solver_class(SpaceMethod.DG,    TimeMethod.BDF,   DgParams())
SolverDgTheta      = make_solver_class(SpaceMethod.DG,    TimeMethod.THETA, DgParams())
SolverLdgBdf       = make_solver_class(SpaceMethod.LDG,   TimeMethod.BDF,   LdgParams())
SolverLdgTheta     = make_solver_class(SpaceMethod.LDG,   TimeMethod.THETA, LdgParams())
SolverPpDgBdf      = make_solver_class(SpaceMethod.PPDG,  TimeMethod.BDF,   PpDgParams())
SolverPpDgTheta    = make_solver_class(SpaceMethod.PPDG,  TimeMethod.THETA, PpDgParams())
SolverSpLdgBdf     = make_solver_class(SpaceMethod.SPLDG, TimeMethod.BDF,   SpLdgParams()) # default reduced
SolverSpLdgTheta   = make_solver_class(SpaceMethod.SPLDG, TimeMethod.THETA, SpLdgParams()) # default reduced