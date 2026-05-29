# wdb-ml

White dwarf binary simulation workflows for generating synthetic ZTF/SCoPe-like
lightcurves, running SCoPe feature generation, and classifying the simulated
sources with the SCoPe XGB and DNN models.

This repository is an umbrella project. The main `wdb-ml` code owns the
simulation notebooks, queue/export scripts, and local path helpers. The external
packages remain standalone installable projects under `external/`.

## Repository Layout

```text
wdb-ml/
  notebooks/
    scope_gaia_crossmatch.ipynb
    simulate_lcurve_wdbs.ipynb
    export_simulated_lcs_for_scope.ipynb
  scripts/
    run_simulated_scope_features.py
    combine_simulated_scope_preds.py
  src/wdb_ml/
    paths.py
    bootstrap.py
    inference_preflight.py
  external/
    scope-ml/
    lcurve-rs/
    periodfind/
  data/
  outputs/
```

## Fresh Install

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:joh15016/wdb-ml.git
cd wdb-ml
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate wdb-ml
python -m ipykernel install --user --name wdb-ml --display-name "Python (wdb-ml)"
```

If you already created the environment, install the packages directly:

```bash
pip install -e .
pip install -e external/scope-ml[train]
pip install -e external/lcurve-rs/python
pip install -e external/periodfind
```

Copy the local config template:

```bash
cp config.example.yaml config.yaml
```

SCoPe itself still reads `external/scope-ml/config.yaml`, so credentials and
SCoPe-specific paths should be configured there as well. Do not commit either
`config.yaml` file.

## SCoPe Fork

The `external/scope-ml` submodule should point to a fork of SCoPe that already
contains the WDB compatibility changes needed by this project:

- lazy TensorFlow/XGBoost imports
- legacy Keras loading for saved DNN `.h5` models via `tf-keras`
- metadata alignment after inference row drops
- `feature_stats='config'` handling

After forking SCoPe on GitHub and pushing those changes to your fork's `main`,
point the submodule at that fork and pin the updated commit:

```bash
git submodule set-url external/scope-ml git@github.com:<your-user>/scope-ml.git
git submodule sync external/scope-ml
git -C external/scope-ml checkout main
git -C external/scope-ml pull origin main

git add .gitmodules external/scope-ml
git commit -m "Use wdb-compatible scope-ml fork"
```

New users can then clone `wdb-ml` with `--recurse-submodules` and get the pinned
SCoPe fork commit directly.

## Required Local Assets

The following are intentionally not tracked in Git:

- Gaia WD catalog files
- WD/SCoPe crossmatch parquet files
- simulated lightcurve batches
- SCoPe queue files
- generated feature parquet files
- SCoPe trained XGB/DNN model artifacts
- SCoPe training parquet files
- prediction outputs
- private config files and tokens

Place local assets in `data/` and SCoPe model/training assets under
`external/scope-ml/` where SCoPe expects them.

## Pipeline

Run the workflow in this order:

```text
notebooks/scope_gaia_crossmatch.ipynb
notebooks/simulate_lcurve_wdbs.ipynb
notebooks/export_simulated_lcs_for_scope.ipynb
python scripts/run_simulated_scope_features.py
python scripts/combine_simulated_scope_preds.py
```

Use `--reuse-features` when rerunning inference from an existing feature parquet:

```bash
python scripts/run_simulated_scope_features.py --reuse-features
```

Run only one inference backend if needed:

```bash
python scripts/run_simulated_scope_features.py --reuse-features --algorithms xgb
python scripts/run_simulated_scope_features.py --reuse-features --algorithms dnn
```
