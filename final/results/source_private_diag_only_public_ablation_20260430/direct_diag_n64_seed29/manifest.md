# Source-Private Hidden-Repair Packet Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py --examples 64 --candidates 4 --seed 29 --budgets 2,4,8 --family-set all --output-dir results/source_private_diag_only_public_ablation_20260430/direct_diag_n64_seed29
```

## Outcome

- strict smoke pass: `True`
- passing budgets: `[2, 4, 8]`
- best budget bytes: `2`

## Artifacts

- `benchmark.jsonl`
- `sweep_summary.json`
- `sweep_summary.md`
- `manifest.json`
- `manifest.md`
- `predictions_budget2.jsonl`
- `summary_budget2.json`
- `predictions_budget4.jsonl`
- `summary_budget4.json`
- `predictions_budget8.jsonl`
- `summary_budget8.json`
