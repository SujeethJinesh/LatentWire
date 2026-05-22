# Architecture Overview

The release scaffold separates stable reproduction concerns from experimental
research code:

- `outlier_migrate.config`: YAML config loading.
- `outlier_migrate.data`: prompt and JSON artifact helpers.
- `outlier_migrate.model_adapter`: dry-run adapter interface.
- `outlier_migrate.quantization`: CPU reference W4A16-style helpers.
- `outlier_migrate.methods`: protected-channel selection methods.
- `outlier_migrate.metrics`: recovery and interval summaries.
- `outlier_migrate.analysis`: small report assembly helpers.
- `outlier_migrate.environment`: config and package verification helpers.
- `scripts`: dry-run entry points that write manifests under `results/`.

The scaffold is designed to accept final result packets later without importing
the experimental runners.

## Release Boundaries

The release package does not import from `experimental/`, does not encode final
paper numbers, and does not require model weights for CPU validation. Full
reproduction commands should be layered on these interfaces once final artifacts
are frozen.
