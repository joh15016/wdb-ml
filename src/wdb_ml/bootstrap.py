import sys

from .paths import (
    LCURVE_ROOT,
    LCURVE_PYTHON_ROOT,
    PERIODFIND_ROOT,
    PROJECT_ROOT,
    SCOPE_ML_ROOT,
)


def bootstrap_paths():
    """Add local checkout dependencies to sys.path for notebooks.

    Editable installs are preferred for normal use. This helper keeps notebooks
    runnable before the conda environment has been fully rebuilt.
    """
    for path in [PROJECT_ROOT / "src", SCOPE_ML_ROOT, LCURVE_PYTHON_ROOT, PERIODFIND_ROOT]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return {
        "project": PROJECT_ROOT,
        "scope_ml": SCOPE_ML_ROOT,
        "lcurve": LCURVE_ROOT,
        "lcurve_python": LCURVE_PYTHON_ROOT,
        "periodfind": PERIODFIND_ROOT,
    }
