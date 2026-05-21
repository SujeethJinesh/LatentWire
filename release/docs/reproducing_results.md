# Reproducing Results Template

This template records the release command sequence once final paper artifacts
are frozen.

## Environment

- Python version:
- Package version:
- Hardware:
- CUDA/runtime context:

## Dry-Run Check

```bash
python -m scripts.run_measurement --config configs/granite_small.yaml --dry-run
python -m scripts.run_intervention --config configs/granite_small.yaml --dry-run
python -m scripts.analyze_results --dry-run
```

## Final Result Artifacts

Final artifact names, hashes, and expected metric values are intentionally
pending until paper numbers are frozen.
