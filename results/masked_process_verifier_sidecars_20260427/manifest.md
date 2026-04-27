# Masked Process-Verifier Sidecars

- date: `2026-04-27`
- git commit: `44c2a4af0b4d75033ebbcaf5b26dbf50f0d98d8a`

| Surface | N | Clean | Empty Pool | Answer-Excluded Top | Output |
|---|---:|---:|---:|---:|---|
| `live` | 70 | 6 | 0 | 55 | `results/masked_process_verifier_sidecars_20260427/live_masked_process_sidecars.jsonl` |
| `holdout` | 70 | 2 | 0 | 62 | `results/masked_process_verifier_sidecars_20260427/holdout_masked_process_sidecars.jsonl` |

## Command

```bash
scripts/materialize_masked_process_verifier_sidecars.py --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json --output-dir results/masked_process_verifier_sidecars_20260427 --date 2026-04-27
```
