"""
MPI smoke test.

Launches the existing smoke tests under ``mpirun -n 2`` as a subprocess and
asserts that pytest exits cleanly. No accuracy is checked; the only assertion
is that the full test suite completes without errors in a 2-rank MPI context.

Run with plain pytest (no mpirun needed at the outer level):
    pytest tests/test_mpi_smoke.py -v
"""

import subprocess
import sys
from pathlib import Path

_TESTS_DIR   = Path(__file__).parent
_SMOKE_FILE  = _TESTS_DIR / "test_smoke.py"
_MPI_RANKS   = 2


def test_smoke_under_mpi():
    """Run test_smoke.py under mpirun -n 2 and assert clean exit."""
    result = subprocess.run(
        [
            "mpirun", "-n", str(_MPI_RANKS),
            sys.executable, "-m", "pytest", str(_SMOKE_FILE),
            "-v", "--tb=short",
        ],
        capture_output=True,
        text=True,
    )

    # Print output so it appears in pytest's captured log on failure
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    assert result.returncode == 0, (
        f"pytest under mpirun -n {_MPI_RANKS} exited with code "
        f"{result.returncode}.\nSee captured output above for details."
    )
