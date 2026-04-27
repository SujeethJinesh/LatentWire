# Candidate-Score Sidecar Top Selector

- date: `2026-04-27`
- status: `top_sidecar_selector_fails_smoke`
- git commit: `44c2a4af0b4d75033ebbcaf5b26dbf50f0d98d8a`
- min confidence: `0.5`

| Condition | Correct | Accepted | Clean Correct | Accepted Harm |
|---|---:|---:|---:|---:|
| `matched` | 8/70 | 0 | 0 | 0 |
| `shuffled_source` | 8/70 | 0 | 0 | 0 |
| `random_sidecar` | 9/70 | 1 | 0 | 0 |
| `target_only` | 8/70 | 0 | 0 | 0 |
| `slots_only` | 8/70 | 0 | 0 | 0 |

- source-necessary clean IDs: none
- control clean union IDs: none

## Command

```bash
scripts/analyze_candidate_score_sidecar_top_select.py --target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json --sidecar-jsonl results/masked_process_verifier_sidecars_20260427/holdout_masked_process_sidecars.jsonl --min-confidence 0.5 --min-source-necessary-clean 1 --max-control-clean-union 0 --max-accepted-harm 0 --date 2026-04-27 --output-json results/masked_process_verifier_sidecars_20260427/holdout_top_selector.json --output-md results/masked_process_verifier_sidecars_20260427/holdout_top_selector.md
```
