#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap(project_root: Path):
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from wdb_ml.bootstrap import bootstrap_paths

    paths = bootstrap_paths()
    scope_root = paths["scope_ml"]
    if str(scope_root) not in sys.path:
        sys.path.insert(0, str(scope_root))
    return scope_root


def _require(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _load_parquet(path: Path) -> pd.DataFrame:
    from scope.utils import read_parquet

    return read_parquet(str(path))


def _merge_metadata(xgb: pd.DataFrame, dnn: pd.DataFrame) -> pd.DataFrame:
    shared_cols = [
        col
        for col in xgb.columns
        if col in dnn.columns and not col.endswith("_xgb") and not col.endswith("_dnn")
    ]
    shared_cols = [col for col in shared_cols if col != "_id"]

    check = xgb[["_id"] + shared_cols].merge(
        dnn[["_id"] + shared_cols],
        on="_id",
        how="inner",
        suffixes=("_xgb", "_dnn"),
        validate="one_to_one",
    )

    mismatches = []
    for col in shared_cols:
        left = check[f"{col}_xgb"]
        right = check[f"{col}_dnn"]
        equal = left.eq(right) | (left.isna() & right.isna())
        if not bool(np.all(equal.to_numpy())):
            mismatches.append(col)

    if mismatches:
        raise ValueError(
            "XGB and DNN parquet files disagree on shared metadata columns: "
            + ", ".join(mismatches)
        )

    dnn_classifier_cols = [col for col in dnn.columns if col.endswith("_dnn")]
    combined = xgb.merge(
        dnn[["_id"] + dnn_classifier_cols],
        on="_id",
        how="inner",
        validate="one_to_one",
    )

    combined.attrs = {**xgb.attrs, **dnn.attrs}
    combined.attrs["combined_from"] = {
        "xgb": str(xgb.attrs.get("inference_dateTime_utc", "")),
        "dnn": str(dnn.attrs.get("inference_dateTime_utc", "")),
    }
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine simulated SCoPe XGB and DNN prediction parquets."
    )
    parser.add_argument(
        "--xgb-path",
        default="preds_xgb/simulated_wdb_specific_ids.parquet",
        help="Relative path to the simulated XGB parquet under scope-ml.",
    )
    parser.add_argument(
        "--dnn-path",
        default="preds_dnn/simulated_wdb_specific_ids.parquet",
        help="Relative path to the simulated DNN parquet under scope-ml.",
    )
    parser.add_argument(
        "--output-path",
        default="preds_dnn_xgb/simulated_wdb_specific_ids.parquet",
        help="Relative output path under scope-ml.",
    )
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--do-not-save", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scope_root = _bootstrap(project_root)

    xgb_path = scope_root / args.xgb_path
    dnn_path = scope_root / args.dnn_path
    output_path = scope_root / args.output_path

    _require(xgb_path, "XGB predictions parquet")
    _require(dnn_path, "DNN predictions parquet")

    from scope.utils import write_parquet

    xgb = _load_parquet(xgb_path)
    dnn = _load_parquet(dnn_path)
    combined = _merge_metadata(xgb, dnn)

    print(f"XGB rows: {len(xgb)}, DNN rows: {len(dnn)}, combined rows: {len(combined)}")
    classifier_cols = [col for col in combined.columns if col.endswith("_xgb") or col.endswith("_dnn")]
    print(f"Classifier columns: {len(classifier_cols)}")
    print(combined.head().to_string(index=False))

    if args.do_not_save:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(combined, str(output_path))
    if args.write_csv:
        combined.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
