from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOME_ROOT = PROJECT_ROOT.parent
EXTERNAL_ROOT = PROJECT_ROOT / "external"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


SCOPE_ML_ROOT = _first_existing(
    EXTERNAL_ROOT / "scope-ml",
    PROJECT_ROOT / "scope-ml",
    HOME_ROOT / "scope-ml",
)
LCURVE_ROOT = _first_existing(
    EXTERNAL_ROOT / "lcurve-rs",
    PROJECT_ROOT / "lcurve-rs",
    SCOPE_ML_ROOT / "lcurve-rs",
)
LCURVE_PYTHON_ROOT = LCURVE_ROOT / "python"
PERIODFIND_ROOT = _first_existing(
    EXTERNAL_ROOT / "periodfind",
    PROJECT_ROOT / "periodfind",
    SCOPE_ML_ROOT / "periodfind",
)
