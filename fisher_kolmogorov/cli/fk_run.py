"""
fk-run — run a generic Fisher-Kolmogorov forward simulation.

Usage
-----
    fk-run --config path/to/my_simulation/

The config folder must contain a ``fk_config.py`` file that defines:

    run_config  : RunConfig       — mesh, PDE coefficients, initial condition
    sim_params  : SimulationParams — numerical parameters
    solver_key  : str             — solver name as in the CLI registry
                                    (e.g. "spldg_bdf", "dg_bdf")

Optionally it may also define:

    model_params : ModelParams subclass instance — solver-specific parameters.
                   If absent, the registry default for ``solver_key`` is used.
    linearize    : bool — whether to linearize the reaction term (default False).

Example ``fk_config.py``
------------------------
    from dolfin import *
    from fisher_kolmogorov.configs.test_configs.base import RunConfig, SimulationParams
    from fisher_kolmogorov.configs.test_configs.run  import make_run_config
    from fisher_kolmogorov.utilities.enum_utilities  import PolyDegree, ThetaMethod

    def my_ic(x, t):
        return exp(-10 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

    run_config = make_run_config(
        user_mesh_path = "my_mesh.msh",
        c_0            = my_ic,
        alpha          = 1.0,
        D              = 1e-2,
    )

    sim_params = SimulationParams(
        l         = PolyDegree.P2,
        T         = 10.0,
        dt        = 0.1,
        nu_or_tht = ThetaMethod.CN,
    )

    solver_key = "spldg_bdf"
"""

import argparse
import importlib.util
import os
import sys

from pathlib import Path

from fisher_kolmogorov.models.solver_factory    import make_solver_class
from fisher_kolmogorov.utilities.enum_utilities import TimeMethod
from fisher_kolmogorov.cli._registries          import _SOLVER_REGISTRY, _PARAMS_DEFAULTS
from fisher_kolmogorov.runners.run_simulation   import run_simulation


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fk-run",
        description="Run a generic Fisher-Kolmogorov forward simulation from a config folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the config folder containing fk_config.py.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for XDMF solution export. Overrides any output_dir in the config.",
    )
    return parser.parse_args(argv)


def _load_config_module(config_dir: Path):
    """Dynamically import fk_config.py from config_dir."""
    config_file = config_dir / "fk_config.py"
    if not config_file.exists():
        print(f"[ERROR]: fk_config.py not found in {config_dir}")
        sys.exit(1)

    # Set working directory to config_dir so relative paths in the config work
    os.chdir(config_dir)

    spec   = importlib.util.spec_from_file_location("fk_config", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_attr(module, name, default=None, required=False):
    value = getattr(module, name, default)
    if required and value is None:
        print(f"[ERROR]: '{name}' must be defined in fk_config.py")
        sys.exit(1)
    return value


def main(argv=None):
    args       = _parse_args(argv)
    config_dir = args.config.resolve()

    if not config_dir.is_dir():
        print(f"[ERROR]: Config path is not a directory: {config_dir}")
        sys.exit(1)

    module = _load_config_module(config_dir)

    # Required attributes
    run_config  = _get_attr(module, "run_config",  required=True)
    sim_params  = _get_attr(module, "sim_params",  required=True)
    solver_key  = _get_attr(module, "solver_key",  required=True)

    if solver_key not in _SOLVER_REGISTRY:
        print(f"[ERROR]: Unknown solver_key '{solver_key}'. "
              f"Available: {list(_SOLVER_REGISTRY.keys())}")
        sys.exit(1)

    # Optional attributes
    model_params = _get_attr(module, "model_params",
                             default=_PARAMS_DEFAULTS[solver_key]())
    linearize    = _get_attr(module, "linearize", default=False)

    space, time, full = _SOLVER_REGISTRY[solver_key]
    linearize         = linearize and time is TimeMethod.BDF

    solver_class = make_solver_class(
        space     = space,
        time      = time,
        params    = model_params,
        linearize = linearize,
        full      = full,
    )

    run_simulation(
        solver_class = solver_class,
        run_config   = run_config,
        sim_params   = sim_params,
        output_dir   = args.output_dir,
    )


if __name__ == "__main__":
    main()
