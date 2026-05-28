from __future__ import annotations

import argparse
import importlib.util
import pickle
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
    return paths


def _require(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Run export_simulated_lcs_for_scope.ipynb first to create the simulated queue."
        )


def _load_queue(queue_dir: Path):
    source_list_path = queue_dir / "source_list.parquet"
    local_lightcurves_path = queue_dir / "local_lightcurves.pkl"
    manifest_path = queue_dir / "manifest.parquet"

    _require(source_list_path, "source-list parquet")
    _require(local_lightcurves_path, "local light-curve pickle")
    _require(manifest_path, "manifest parquet")

    source_list = pd.read_parquet(source_list_path)
    manifest = pd.read_parquet(manifest_path)
    with local_lightcurves_path.open("rb") as handle:
        local_lightcurves = pickle.load(handle)

    return source_list, local_lightcurves, manifest


def _safe_int(value, default=0):
    if pd.isna(value):
        return default
    return int(value)


def _discover_model_specs(scope_root: Path, algorithm: str):
    if algorithm == "xgb":
        model_root = scope_root / "models_xgb/trained_xgb_models"
        model_suffix = ".json"
    elif algorithm == "dnn":
        model_root = scope_root / "models_dnn/trained_dnn_models"
        model_suffix = ".h5"
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if not model_root.exists():
        raise FileNotFoundError(f"Missing model root for {algorithm}: {model_root}")

    model_specs = []
    for class_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        model_files = sorted(class_dir.glob(f"*{model_suffix}"))
        if not model_files:
            continue
        model_specs.append((class_dir.name, model_files[-1]))

    if not model_specs:
        raise FileNotFoundError(f"No trained {algorithm} models found in {model_root}")

    model_class_names = [name for name, _ in model_specs]
    paths_models = [path for _, path in model_specs]
    return model_class_names, paths_models


def _algorithm_output_prefix(base_output: str, algorithm: str) -> str:
    output_path = Path(base_output)
    parts = list(output_path.parts)

    for index, part in enumerate(parts):
        if part.startswith("preds_"):
            parts[index] = f"preds_{algorithm}"
            return str(Path(*parts))

    return str(Path(f"preds_{algorithm}") / output_path)


def build_local_xmatch(manifest: pd.DataFrame, sim_meta_path: Path, matches_path: Path):
    """Build a local external-catalog cache for simulated sources.

    SCoPe inference expects external-catalog columns to exist even when the
    model class does not use them. For simulated lightcurves, inherit Gaia
    metadata from the real WD match used as the cadence/template and fill
    unavailable AllWISE/PS1 measurements with nulls so SCoPe can impute them.
    """
    sim_meta = pd.read_parquet(sim_meta_path) if sim_meta_path.exists() else pd.DataFrame()
    matches = pd.read_parquet(matches_path) if matches_path.exists() else pd.DataFrame()

    match_by_source = {}
    if not matches.empty and "_id" in matches.columns:
        for _, row in matches.iterrows():
            if pd.notna(row.get("_id")):
                match_by_source[_safe_int(row["_id"])] = row

    meta_by_sim_id = {}
    if not sim_meta.empty and "sim_id" in sim_meta.columns:
        meta_by_sim_id = {row["sim_id"]: row for _, row in sim_meta.iterrows()}

    external_feature_columns = [
        "AllWISE__w1mpro",
        "AllWISE__w1sigmpro",
        "AllWISE__w2mpro",
        "AllWISE__w2sigmpro",
        "AllWISE__w3mpro",
        "AllWISE__w4mpro",
        "Gaia_EDR3__parallax",
        "Gaia_EDR3__parallax_error",
        "Gaia_EDR3__phot_bp_mean_mag",
        "Gaia_EDR3__phot_bp_rp_excess_factor",
        "Gaia_EDR3__phot_g_mean_mag",
        "Gaia_EDR3__phot_rp_mean_mag",
        "PS1_DR1__gMeanPSFMag",
        "PS1_DR1__gMeanPSFMagErr",
        "PS1_DR1__iMeanPSFMag",
        "PS1_DR1__iMeanPSFMagErr",
        "PS1_DR1__rMeanPSFMag",
        "PS1_DR1__rMeanPSFMagErr",
        "PS1_DR1__yMeanPSFMag",
        "PS1_DR1__yMeanPSFMagErr",
        "PS1_DR1__zMeanPSFMag",
        "PS1_DR1__zMeanPSFMagErr",
    ]

    local_xmatch = {}
    for _, manifest_row in manifest.iterrows():
        ztf_id = _safe_int(manifest_row["ztf_id"])
        sim_id = manifest_row.get("sim_id")
        meta_row = meta_by_sim_id.get(sim_id)
        match_row = None
        if meta_row is not None and pd.notna(meta_row.get("source_id")):
            match_row = match_by_source.get(_safe_int(meta_row["source_id"]))

        row = {column: np.nan for column in external_feature_columns}
        row.update({"AllWISE___id": 0, "PS1_DR1___id": 0, "Gaia_EDR3___id": 0})

        if match_row is not None:
            for column in row:
                if column in match_row.index and pd.notna(match_row[column]):
                    row[column] = match_row[column]

        if meta_row is not None and pd.notna(meta_row.get("gaia_source_id")):
            row["Gaia_EDR3___id"] = _safe_int(meta_row["gaia_source_id"])

        local_xmatch[ztf_id] = row

    return local_xmatch


def main():
    parser = argparse.ArgumentParser(
        description="Generate SCoPe features and classifier inference for simulated WDB lightcurves."
    )
    parser.add_argument(
        "--queue-dir",
        default="data/scope_simulated_wdb_queue",
        help="Directory containing source_list.parquet, local_lightcurves.pkl, and manifest.parquet.",
    )
    parser.add_argument(
        "--feature-dir",
        default="generated_features_simulated_wdb",
        help="Output directory, relative to scope-ml, for generated feature parquet.",
    )
    parser.add_argument(
        "--feature-prefix",
        default="gen_features_simulated_wdb",
        help="Prefix for generated feature parquet.",
    )
    parser.add_argument(
        "--pred-output",
        default="preds_xgb_simulated_wdb/simulated_wdb_specific_ids",
        help="Output path prefix, relative to scope-ml, for inference predictions.",
    )
    parser.add_argument(
        "--sim-meta",
        default="data/simulated_wdb_batch/sim_meta.parquet",
        help="Saved simulation metadata parquet, used to inherit source metadata.",
    )
    parser.add_argument(
        "--matches-path",
        default="data/random_100_gaia_wd_scope_nonvariable.parquet",
        help="Original WD/SCoPe match parquet, used as a local xmatch cache.",
    )
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--ncore", type=int, default=1)
    parser.add_argument("--period-batch-size", type=int, default=128)
    parser.add_argument("--min-n-lc-points", type=int, default=30)
    parser.add_argument("--min-cadence-minutes", type=float, default=5.0)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("xgb", "dnn"),
        default=("xgb", "dnn")
        if __import__("importlib").util.find_spec("tensorflow") is not None
        else ("xgb",),
        help="Inference algorithms to run; dnn is skipped automatically if TensorFlow is unavailable.",
    )
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Skip feature generation if the expected feature parquet already exists.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    paths = _bootstrap(project_root)
    scope_root = paths["scope_ml"]

    queue_dir = project_root / args.queue_dir
    source_list, local_lightcurves, manifest = _load_queue(queue_dir)
    print(f"Loaded {len(source_list)} sources, {len(local_lightcurves)} lightcurves.")
    print(f"Manifest rows: {len(manifest)}")

    feature_file = (
        project_root
        / args.feature_dir
        / "specific_ids"
        / f"{args.feature_prefix}_specific_ids.parquet"
    )

    local_alert_stats = {
        int(lc["_id"]): {"n_ztf_alerts": 0, "mean_ztf_alert_braai": 0.0}
        for lc in local_lightcurves
    }
    local_xmatch = build_local_xmatch(
        manifest=manifest,
        sim_meta_path=project_root / args.sim_meta,
        matches_path=project_root / args.matches_path,
    )

    if args.reuse_features and feature_file.exists():
        print(f"Reusing existing feature file: {feature_file}")
        feature_df = pd.read_parquet(feature_file)
    else:
        from tools.generate_features import generate_features

        feature_df = generate_features(
            doSpecificIDs=True,
            fg_dataset=str(queue_dir / "source_list.parquet"),
            local_lightcurves=local_lightcurves,
            local_alert_stats=local_alert_stats,
            local_xmatch=local_xmatch,
            skipCloseSources=True,
            doCPU=True,
            period_algorithms=["ELS_ECE_EAOV"],
            period_batch_size=args.period_batch_size,
            doNotSave=False,
            dirname=args.feature_dir,
            filename=args.feature_prefix,
            min_n_lc_points=args.min_n_lc_points,
            min_cadence_minutes=args.min_cadence_minutes,
            limit=args.limit,
            Ncore=args.ncore,
        )
        print(f"Generated features: {feature_df.shape}")

    _require(feature_file, "generated feature parquet")

    required = [
        "period_ELS_ECE_EAOV",
        "f1_amp_ELS_ECE_EAOV",
        "f1_BIC_ELS_ECE_EAOV",
    ]
    missing = [column for column in required if column not in feature_df.columns]
    if missing:
        raise RuntimeError(f"Feature generation did not create required columns: {missing}")

    if args.skip_inference:
        print(f"Feature file ready: {feature_file}")
        return

    from wdb_ml.inference_preflight import check_inference_inputs

    cwd = Path.cwd()
    try:
        import os

        os.chdir(scope_root)
        import scope.utils as scope_utils
        import tools.inference as inference

        # scope.utils may have been imported earlier while cwd was project_root.
        # Force inference-time config/training paths to resolve from scope-ml.
        scope_utils.BASE_DIR = scope_root
        inference.BASE_DIR = scope_root
        inference.BASE_DIR_PREDS = scope_root
        inference.BASE_DIR_FEATS = project_root
        inference.TRAINING_SET = inference.read_parquet(
            scope_root / "tools/fritzDownload/merged_classifications_features.parquet"
        )

        for algorithm in args.algorithms:
            if algorithm == "dnn":
                if importlib.util.find_spec("tensorflow") is None:
                    print("Skipping dnn: TensorFlow is not installed in the active environment.")
                    continue
                import os

                os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

            model_class_names, paths_models = _discover_model_specs(scope_root, algorithm)
            report = check_inference_inputs(
                feature_file,
                model_class_names=model_class_names,
                paths_models=paths_models,
                algorithm=algorithm,
                period_suffix="ELS_ECE_EAOV",
            )
            print(f"{algorithm} preflight:", report.as_dict())
            if not report.ok:
                raise RuntimeError(f"Inference preflight failed for {algorithm}.")

            preds, outfile = inference.run_inference(
                paths_models=[str(path) for path in paths_models],
                model_class_names=model_class_names,
                field="specific_ids",
                xgb_model=(algorithm == "xgb"),
                feature_directory=args.feature_dir,
                feature_file_prefix=args.feature_prefix,
                features_filename=str(feature_file),
                period_suffix="ELS_ECE_EAOV",
                no_write_metadata=True,
                output=_algorithm_output_prefix(args.pred_output, algorithm),
                batch_size=10000,
            )

            prediction_columns = [column for column in preds.columns if column.endswith(f"_{algorithm}")]
            print(f"{algorithm} predictions written to: {outfile}")
            print(f"{algorithm} classifier columns ({len(prediction_columns)}): {prediction_columns}")
            print(preds[["_id", "period"] + prediction_columns].head().to_string(index=False))
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
