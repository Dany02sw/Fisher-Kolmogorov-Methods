"""
Shared launcher for examples/main scripts.

Import this module from any examples/main/{solver}.py and call ``launch``.
Thin wrapper around ``fisher_kolmogorov.cli._run_core.launch`` — all logic
lives there so it is shared with the ``fk-convergence`` CLI entry point.
"""

from fisher_kolmogorov.cli._run_core import launch  # noqa: F401