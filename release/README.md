# OutlierMigrate Release Scaffold

This directory contains the minimal release-facing reproducibility framework for
OutlierMigrate. It is intentionally separate from `experimental/` and does not
copy experimental runners.

The current scaffold provides:

- typed configuration and prompt-loading utilities;
- CPU-only model adapter stubs for dry-run reproduction flows;
- simple symmetric W4A16-style quantization helpers;
- method registry and placeholder method families;
- metric and analysis helpers;
- dry-run command-line entry points that write manifests under `./results/`;
- CPU tests for quantization, methods, metrics, and registry extension.

Final paper numbers, frozen result packets, and full model execution are not
encoded here yet. Those should be added only after the paper-facing artifacts
are finalized.

## Quick Start

```bash
cd release
python -m pip install -e ".[dev]"
pytest
python -m scripts.run_measurement --config configs/granite_small.yaml --dry-run
python -m scripts.run_intervention --config configs/granite_small.yaml --dry-run
python -m scripts.analyze_results --dry-run
```

All CLIs default to writing under `./results/` relative to the current working
directory.
