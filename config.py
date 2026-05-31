from pathlib import Path

ROOT_DIR            = Path(__file__).parent.resolve()
DEFAULT_RESULTS_DIR = ROOT_DIR / "Results" / "Default"
CONVERGENCE_DIR     = ROOT_DIR / "Results" / "Convergence"
MESHES_DIR          = ROOT_DIR / "Results" / "Meshes"
WAVES_DIR           = ROOT_DIR / "Results" / "Simulations" / "Waves"
BRAIN_DIR           = ROOT_DIR / "Results" / "Simulations" / "Brain"

SAGITTAL_DIR   = BRAIN_DIR / "Sagittal"
CORONAL_DIR    = BRAIN_DIR / "Coronal"
HORIZONTAL_DIR = BRAIN_DIR / "Horizontal"


# Functions to create all directories if they do not exists _______________________________________________________________________________
def get_convergence_dir(space_method, time_method):
    """Returns the convergence output directory for a given space+time method pair."""
    return CONVERGENCE_DIR / f"{space_method.name}{time_method.name}"

def get_waves_dir(space_method, time_method, l, time_order):
    """Returns the waves output directory for a given space+time method pair and polynomial degree."""
    return WAVES_DIR / f"{space_method.name}{time_method.name}" / f"P{l}_time_order_{time_order}"

def ensure_waves_dir(space_method, time_method, l, time_order):
    """Creates and returns the waves output directory for a given method pair and polynomial degree."""
    d = get_waves_dir(space_method, time_method, l, time_order)
    d.mkdir(parents=True, exist_ok=True)
    return d

def ensure_dirs(space_method=None, time_method=None):
    """Creates all output directories. If space+time methods are provided, also creates their convergence and waves subdirectories."""
    dirs = [MESHES_DIR, SAGITTAL_DIR, CORONAL_DIR, HORIZONTAL_DIR, DEFAULT_RESULTS_DIR]
    if space_method is not None and time_method is not None:
        dirs.append(get_convergence_dir(space_method, time_method))
        dirs.append(get_waves_dir(space_method, time_method))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Executed at import time
ensure_dirs()