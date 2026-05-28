from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .paths import PROJECT_ROOT, SCOPE_ML_ROOT


BASE_METADATA_COLUMNS = [
    "_id",
    "ra",
    "dec",
    "field",
    "ccd",
    "quad",
    "Gaia_EDR3___id",
    "AllWISE___id",
    "PS1_DR1___id",
]


@dataclass
class InferencePreflightReport:
    feature_file: Path
    row_count: int
    algorithm: str
    model_class_names: list[str]
    missing_columns: list[str]
    missing_model_paths: list[Path]
    missing_training_set: Path | None
    missing_feature_stats: list[str]
    nullable_required_columns: list[str]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_columns
            or self.missing_model_paths
            or self.missing_training_set
            or self.missing_feature_stats
        )

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "feature_file": str(self.feature_file),
            "row_count": self.row_count,
            "algorithm": self.algorithm,
            "model_class_names": self.model_class_names,
            "missing_columns": self.missing_columns,
            "missing_model_paths": [str(path) for path in self.missing_model_paths],
            "missing_training_set": (
                None if self.missing_training_set is None else str(self.missing_training_set)
            ),
            "missing_feature_stats": self.missing_feature_stats,
            "nullable_required_columns": self.nullable_required_columns,
        }


def _load_scope_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else SCOPE_ML_ROOT / "config.yaml"
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1"}
    return bool(value)


def _feature_columns_for_class(
    config: dict,
    model_class_name: str,
    period_suffix: str | None,
    algorithm: str,
) -> list[str]:
    train_config = config["training"]["classes"][model_class_name]
    feature_group = train_config["features"]
    feature_config = config["features"][feature_group]

    names: list[str] = []
    for name, info in feature_config.items():
        if not _truthy(info.get("include", False)):
            continue
        if info.get("periodic", False) and period_suffix not in {None, "None"}:
            names.append(f"{name}_{period_suffix}")
        else:
            names.append(name)

    if algorithm == "xgb" and "dmdt" in names:
        names.remove("dmdt")
    return names


def _period_column(period_suffix: str | None) -> str:
    if period_suffix in {None, "None"}:
        return "period"
    return f"period_{period_suffix}"


def _resolve_existing_path(path: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def check_inference_inputs(
    feature_file: str | Path,
    model_class_names: Iterable[str] = ("vnv",),
    paths_models: Iterable[str | Path] = (),
    algorithm: str = "xgb",
    period_suffix: str | None = None,
    config_path: str | Path | None = None,
    training_set: str | Path | pd.DataFrame = "use_config",
    feature_stats: str | dict | None = None,
    sample_rows: int | None = None,
) -> InferencePreflightReport:
    """Check whether a feature parquet is ready for SCoPe inference.

    This intentionally avoids importing ``tools.inference`` so it can run in
    environments that do not have TensorFlow installed yet.
    """
    algorithm = algorithm.lower()
    if algorithm not in {"xgb", "dnn"}:
        raise ValueError("algorithm must be either 'xgb' or 'dnn'")

    config = _load_scope_config(config_path)
    if period_suffix is None:
        period_suffix = config["features"]["info"].get("period_suffix")

    feature_file = _resolve_existing_path(feature_file)
    if not feature_file.exists():
        raise FileNotFoundError(feature_file)

    df = pd.read_parquet(feature_file)
    if sample_rows is not None:
        df = df.head(sample_rows)

    model_class_names = list(model_class_names)
    required_columns = set(BASE_METADATA_COLUMNS)
    required_columns.add(_period_column(period_suffix))
    if "filter" in df.columns:
        required_columns.add("filter")

    for model_class_name in model_class_names:
        required_columns.update(
            _feature_columns_for_class(
                config=config,
                model_class_name=model_class_name,
                period_suffix=period_suffix,
                algorithm=algorithm,
            )
        )

    if algorithm == "dnn":
        required_columns.add("dmdt")

    missing_columns = sorted(required_columns.difference(df.columns))
    nullable_required_columns = sorted(
        column for column in required_columns.intersection(df.columns) if df[column].isna().any()
    )

    missing_model_paths = []
    for path in paths_models:
        resolved = _resolve_existing_path(path)
        if not resolved.exists():
            missing_model_paths.append(resolved)

    missing_feature_stats: list[str] = []
    missing_training_set = None
    if feature_stats == "config":
        stats = config.get("feature_stats") or {}
        missing_feature_stats = sorted(
            column
            for column in required_columns.intersection(df.columns)
            if column not in stats and column not in BASE_METADATA_COLUMNS
        )
    elif isinstance(feature_stats, dict):
        missing_feature_stats = sorted(
            column
            for column in required_columns.intersection(df.columns)
            if column not in feature_stats and column not in BASE_METADATA_COLUMNS
        )
    else:
        if isinstance(training_set, pd.DataFrame):
            pass
        elif training_set == "use_config":
            resolved = _resolve_existing_path(config["training"]["dataset"], SCOPE_ML_ROOT)
            if not resolved.exists():
                missing_training_set = resolved
        elif training_set is not None:
            resolved = _resolve_existing_path(training_set)
            if not resolved.exists():
                missing_training_set = resolved

    return InferencePreflightReport(
        feature_file=feature_file,
        row_count=len(df),
        algorithm=algorithm,
        model_class_names=model_class_names,
        missing_columns=missing_columns,
        missing_model_paths=missing_model_paths,
        missing_training_set=missing_training_set,
        missing_feature_stats=missing_feature_stats,
        nullable_required_columns=nullable_required_columns,
    )
