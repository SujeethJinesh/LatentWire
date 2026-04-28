# Source-Private Hidden-Repair Packet Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py --examples 64 --candidates 4 --seed 28 --budgets 2,4,8,16,32 --output-dir results/source_private_hidden_repair_packet_smoke_20260428
```

## Outcome

- strict smoke pass: `True`
- passing budgets: `[2, 4, 8, 16, 32]`
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
- `predictions_budget16.jsonl`
- `summary_budget16.json`
- `predictions_budget32.jsonl`
- `summary_budget32.json`
