"""
Brain simulation — coronal section.

Usage:
    python3 examples/reproduce/py/brain/_brain_coronal.py
"""

from fisher_kolmogorov.models.solver_factory      import make_solver_class
from fisher_kolmogorov.configs.model_configs      import SpLdgParams
from fisher_kolmogorov.configs.test_configs.base  import SimulationParams
from fisher_kolmogorov.configs.test_configs.brain import make_brain_config
from fisher_kolmogorov.utilities.enum_utilities   import BrainSection, PolyDegree, BdfOrder, SpaceMethod, TimeMethod
from fisher_kolmogorov.runners.brain              import run_brain_simulation

if __name__ == "__main__":
    run_brain_simulation(
        solver_class   = make_solver_class(
            space  = SpaceMethod.SPLDG,
            time   = TimeMethod.BDF,
            params = SpLdgParams(eta_0=2.0, theta=0.5, smoothing=1e-9),
            full   = False,
        ),
        sim_config   = make_brain_config(BrainSection.CORONAL),
        sim_params   = SimulationParams(
            l         = PolyDegree.P2,
            T         = 50.0,
            dt        = 0.25,
            nu_or_tht = BdfOrder.BDF2,
            tol       = 1e-6,
            max_it    = 500,
        ),
    )
