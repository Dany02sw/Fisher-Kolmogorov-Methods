"""
Brain simulation — horizontal section.

Usage:
    python examples/reproduce/py/brain/_brain_horizontal.py
"""

from fisher_kolmogorov.models.solver_spldg_bdf_red2 import SolverSpLdgBDFReduced2
from fisher_kolmogorov.configs.model_configs        import SpLdgParams
from fisher_kolmogorov.configs.test_configs.base    import SimulationParams
from fisher_kolmogorov.configs.test_configs.brain   import make_brain_config
from fisher_kolmogorov.utilities.enum_utilities     import BrainSection, PolyDegree, BdfOrder
from fisher_kolmogorov.runners.brain                import run_brain_simulation

if __name__ == "__main__":
    run_brain_simulation(
        solver_class = SolverSpLdgBDFReduced2,
        model_params = SpLdgParams(eta_0=1.0, theta=-1.0),
        sim_config   = make_brain_config(BrainSection.HORIZONTAL),
        sim_params   = SimulationParams(
            l         = PolyDegree.P2,
            T         = 50.0,
            dt        = 0.25,
            nu_or_tht = BdfOrder.BDF2,
            tol       = 1e-6,
            max_it    = 500,
        ),
    )
