"""
Run all reproduce studies in the order suggested by the paper.

Checkpoint files under reproduce/runs/<study>/.done/ ensure that
already-completed convergence runs are skipped. Safe to interrupt
and relaunch at any time.

The brain simulation has no checkpoint — it runs from scratch each time.

Usage:
    python3 examples/reproduce/run_all_reproduce.py
"""

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent

_CONVERGENCE_STUDIES = [
    "py/cosine/cosine_spatial_all.py",
    "py/cosine/cosine_polynomial.py",
    "py/cosine/cosine_temporal_all.py",
    "py/wave/wave_spatial_saturation_all.py",
    "py/wave/wave_polynomial_saturation_all.py",
]

_BRAIN_STUDY = "py/brain/brain_all.py"


def _run(script: Path) -> None:
    print(f"\n{'='*60}\n  {script.name}\n{'='*60}\n")
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        print(f"\n[error] {script.name} exited with code {result.returncode}.")
        print("Fix the issue and rerun — completed convergence steps will be skipped.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    for study in _CONVERGENCE_STUDIES:
        _run(_HERE / study)
    _run(_HERE / _BRAIN_STUDY)
