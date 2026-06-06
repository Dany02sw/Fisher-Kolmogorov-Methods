"""
Run all reproduce studies in the order suggested by the paper.

Checkpoint files in reproduce/.done/ ensure that already-completed runs
are skipped. Safe to interrupt and relaunch at any time.

Usage:
    python reproduce/run_all.py
"""

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent

_STUDIES = [
    "cosine/cosine_spatial_all.py",
    "cosine/cosine_polynomial.py",
    "cosine/cosine_temporal_all.py",
    "wave/wave_spatial_saturation_all.py",
    "wave/wave_polynomial_saturation_all.py",
]

if __name__ == "__main__":
    for study in _STUDIES:
        script = _HERE / study
        print(f"\n{'='*60}")
        print(f"  {study}")
        print(f"{'='*60}\n")
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            print(f"\n[error] {study} exited with code {result.returncode}.")
            print("Fix the issue and rerun — completed steps will be skipped.")
            sys.exit(result.returncode)
