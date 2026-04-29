# Source-Private Coded-Label Risk Gate Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_coded_label_risk_gate.py --examples 160 --candidates 4 --family-set all --seeds 29,31,37 --budget 2 --output-dir results/source_private_coded_label_risk_gate_20260429
```

## Outcome

- pass gate: `True`
- examples per seed: `160`
- transforms: `['baseline', 'label_rename', 'diagnostic_code_remap', 'candidate_pool_permutation', 'label_code_order_composed']`

## Artifacts

- `summary.json`
- `summary.md`
- `predictions.jsonl`
- `manifest.json`
- `manifest.md`
