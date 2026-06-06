"""
Shared launcher for reproduce/ scripts.

Wraps ``examples._run.launch`` with the fixed configurations used in the paper/report.
Each reproduce script imports ``launch_reproduce`` and calls it with the desired
solver parameters; all tuning knobs (mesh sequences, dt, tolerances, …) live here.

Output capture
--------------
Each call to ``launch_reproduce`` captures everything printed to stdout during
the run and appends it to a per-study log file inside ``examples/reproduce/results/``.
The checkpoint is written only after the append succeeds, so interrupted runs
leave the log in a consistent state and resume by appending to it.
"""

import io
import sys
import unittest.mock

from contextlib import contextmanager
from datetime   import datetime
from pathlib    import Path
from typing     import Optional

from fisher_kolmogorov.utilities.enum_utilities  import ConvType, TestType


# ---------------------------------------------------------------------------
# Paper-fixed run parameters
# ---------------------------------------------------------------------------

#: Nonlinear solver tolerance used in all reproduce runs.
TOL = 1e-11

#: Maximum nonlinear iterations used in all reproduce runs.
MAX_IT = 200

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

_REPRODUCE_DIR = Path(__file__).parent
_RUNS_DIR      = _REPRODUCE_DIR / "runs"

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _study_dir(conv_type: ConvType, test_type: TestType) -> Path:
    """Return the per-study directory, which holds both checkpoints and the log."""
    name = f"{test_type.name.lower()}_{conv_type.name.lower()}"
    return _RUNS_DIR / name


def _mark_done(conv_type: ConvType, test_type: TestType, tag: str) -> None:
    done_dir = _study_dir(conv_type, test_type) / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / tag).touch()


def _is_done(conv_type: ConvType, test_type: TestType, tag: str) -> bool:
    return (_study_dir(conv_type, test_type) / ".done" / tag).exists()


def clear_done(conv_type: ConvType, test_type: TestType) -> None:
    """
    Remove all checkpoint files for the given study, leaving the log intact.

    Call this at the end of a ``*_all`` script so the study can be rerun
    from scratch without manually deleting checkpoint files.
    """
    import shutil
    done_dir = _study_dir(conv_type, test_type) / ".done"
    if done_dir.exists():
        shutil.rmtree(done_dir)


def checkpoint_tag(conv_type: ConvType, test_type: TestType,
                   l: Optional[int] = None, nu: Optional[int] = None) -> str:
    """Build a unique tag for a single (l, nu, conv_type, test_type) run."""
    parts = [test_type.name.lower(), conv_type.name.lower()]
    if l  is not None: parts.append(f"l{l}")
    if nu is not None: parts.append(f"bdf{nu}")
    return "_".join(parts)


def _study_log_path(conv_type: ConvType, test_type: TestType) -> Path:
    """Return the path to the per-study aggregated log file."""
    name = f"{test_type.name.lower()}_{conv_type.name.lower()}.txt"
    return _study_dir(conv_type, test_type) / name


# ---------------------------------------------------------------------------
# Output capture
# ---------------------------------------------------------------------------

@contextmanager
def _capture_and_tee(log_path: Path, tag: str):
    """
    Context manager that tees stdout to both the terminal and an in-memory
    buffer. On clean exit the buffer is appended to log_path with a header.

    Parameters
    ----------
    log_path : Path
        Aggregated log file for the current study.
    tag : str
        Checkpoint tag, used in the section header written to the log.
    """
    buffer   = io.StringIO()
    original = sys.stdout

    class _Tee:
        def write(self, data):
            original.write(data)
            buffer.write(data)
        def flush(self):
            original.flush()

    sys.stdout = _Tee()
    try:
        yield
    finally:
        sys.stdout = original

    captured = buffer.getvalue()
    if captured.strip():
        # Keep only from the timer summary line onward; fall back to full
        # output if the marker is not found (e.g. the run raised an exception).
        _TIMER_MARKER = "elapsed time:"
        idx = captured.find(_TIMER_MARKER)
        if idx != -1:
            line_start = captured.rfind("\n", 0, idx) + 1
            to_log = captured[line_start:]
        else:
            to_log = captured

        with log_path.open("a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'#'*80}\n")
            f.write(f"# {tag}  [{timestamp}]\n")
            f.write(f"{'#'*80}\n")
            f.write(to_log)


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------

def launch_reproduce(solver_class, model_params, conv_type: ConvType,
                     test_type: TestType, l: Optional[int] = None,
                     nu: Optional[int] = None, **solver_kwargs) -> None:
    """
    Run a single reproduce step, skipping it if already completed.

    Stdout produced during the run is tee'd to the terminal and appended to
    the per-study log file in ``reproduce/results/``. The checkpoint is written
    only after the log append succeeds. On resume, new steps are appended to
    the existing log so the file grows monotonically.

    Parameters
    ----------
    solver_class  : type
        Concrete solver class (must be a subclass of one of the 8 solvers).
    model_params  : ModelParams
        Parameter dataclass instance matching the solver.
    conv_type     : ConvType
        Study to run: SPATIAL, POLYNOMIAL, or TEMPORAL.
    test_type     : TestType
        Physical test case: COSINE or WAVE.
    l             : int, optional
        Polynomial degree (used in tag and printed header).
    nu            : int, optional
        BDF order (used in tag and printed header).
    **solver_kwargs
        Extra keyword arguments forwarded to the runner (e.g. Linearize=False).
    """
    from examples._run import launch  # imported here to avoid circular imports

    tag      = checkpoint_tag(conv_type, test_type, l=l, nu=nu)
    _study_dir(conv_type, test_type).mkdir(parents=True, exist_ok=True)
    log_path = _study_log_path(conv_type, test_type)

    if _is_done(conv_type, test_type, tag):
        print(f"[skip] {tag} — already completed")
        return

    with _capture_and_tee(log_path, tag):
        with unittest.mock.patch("matplotlib.pyplot.show"):
            launch(
                solver_class  = solver_class,
                model_params  = model_params,
                conv_type     = conv_type,
                test_type     = test_type,
                tol           = TOL,
                max_it        = MAX_IT,
                l             = l,
                nu            = nu,
                **solver_kwargs,
            )

    _mark_done(conv_type, test_type, tag)
