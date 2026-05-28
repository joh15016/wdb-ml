"""Utilities for the local WDB-ML project layout."""

from .paths import (
    PROJECT_ROOT,
    SCOPE_ML_ROOT,
    LCURVE_ROOT,
    LCURVE_PYTHON_ROOT,
    PERIODFIND_ROOT,
)
from .bootstrap import bootstrap_paths

__all__ = [
    "PROJECT_ROOT",
    "SCOPE_ML_ROOT",
    "LCURVE_ROOT",
    "LCURVE_PYTHON_ROOT",
    "PERIODFIND_ROOT",
    "bootstrap_paths",
]
