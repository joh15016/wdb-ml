# External Packages

This directory is reserved for standalone package checkouts used by `wdb-ml`.

Expected layout:

```text
external/
  scope-ml/
  lcurve-rs/
  periodfind/
```

Use Git submodules for reproducible installs, or manually clone the packages here
before installing them in editable mode.

`scope-ml` should point to a fork containing the WDB inference compatibility
changes required by the simulated inference pipeline.
