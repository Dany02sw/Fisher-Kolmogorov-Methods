from pathlib import Path

ROOT_DIR        = Path(__file__).parent.resolve()
CONVERGENCE_DIR = ROOT_DIR / "Results" / "Convergence"
MESHES_DIR      = ROOT_DIR / "Results" / "Meshes"
WAVES_DIR       = ROOT_DIR / "Results" / "Simulations" / "Waves"
BRAIN_DIR       = ROOT_DIR / "Results" / "Simulations" / "Brain"

SAGITTAL_DIR   = BRAIN_DIR / "Sagittal"
CORONAL_DIR    = BRAIN_DIR / "Coronal"
HORIZONTAL_DIR = BRAIN_DIR / "Horizontal"

# Functions to create all directories if they do not exists _______________________________________________________________________________
def get_convergence_dir(space_method, time_method):
    """Returns the convergence output directory for a given space+time method pair."""
    return CONVERGENCE_DIR / f"{space_method.name}{time_method.name}"


def ensure_dirs(space_method=None, time_method=None):
    """Creates all output directories. If space+time methods are provided, also creates their convergence subdirectory."""
    dirs = [MESHES_DIR, SAGITTAL_DIR, CORONAL_DIR, HORIZONTAL_DIR]
    if space_method is not None and time_method is not None:
        dirs.append(get_convergence_dir(space_method, time_method))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Executed at import time
ensure_dirs()