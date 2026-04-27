# Candidate-Score Sidecar Top Selector

- date: `2026-04-27`
- status: `top_sidecar_selector_passes_smoke`
- git commit: `afbf022b1e7e1e96c1bd76f72c28e6f538f73abf`
- min confidence: `2.0`

| Condition | Correct | Accepted | Clean Correct | Accepted Harm |
|---|---:|---:|---:|---:|
| `matched` | 1/3 | 1 | 1 | 0 |
| `shuffled_source` | 0/3 | 1 | 0 | 0 |
| `random_sidecar` | 0/3 | 1 | 0 | 0 |
| `target_only` | 0/3 | 0 | 0 | 0 |
| `slots_only` | 0/3 | 0 | 0 | 0 |

- source-necessary clean IDs: `14bfbfc94f2c2e7b`
- control clean union IDs: none

## Command

```bash
scripts/analyze_candidate_score_sidecar_top_select.py --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl --min-confidence 2.0 --min-source-necessary-clean 1 --max-control-clean-union 0 --max-accepted-harm 0 --date 2026-04-27 --output-json results/target_only_sampling_clean3_20260427/top_sidecar_selector.json --output-md results/target_only_sampling_clean3_20260427/top_sidecar_selector.md
```
