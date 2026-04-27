# Candidate-Score Sidecar Top Selector

- date: `2026-04-27`
- status: `top_sidecar_selector_fails_smoke`
- git commit: `6a09be6d62e3804af408cad178b8274b965b7da6`
- min confidence: `2.0`

| Condition | Correct | Accepted | Clean Correct | Accepted Harm |
|---|---:|---:|---:|---:|
| `matched` | 0/3 | 0 | 0 | 0 |
| `shuffled_source` | 0/3 | 0 | 0 | 0 |
| `random_sidecar` | 0/3 | 0 | 0 | 0 |
| `target_only` | 0/3 | 0 | 0 | 0 |
| `slots_only` | 0/3 | 0 | 0 | 0 |

- source-necessary clean IDs: none
- control clean union IDs: none

## Command

```bash
scripts/analyze_candidate_score_sidecar_top_select.py --target-set results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_target_set.json --sidecar-jsonl results/qwen25math7b_svamp70_target_sampling_clean3_20260427/source_candidate_sidecars_answer_masked/live_candidate_sidecars.jsonl --min-confidence 2.0 --min-source-necessary-clean 1 --max-control-clean-union 0 --max-accepted-harm 0 --date 2026-04-27 --output-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/top_selector_answer_masked.json --output-md results/qwen25math7b_svamp70_target_sampling_clean3_20260427/top_selector_answer_masked.md
```
