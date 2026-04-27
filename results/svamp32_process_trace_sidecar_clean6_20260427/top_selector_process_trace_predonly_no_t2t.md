# Candidate-Score Sidecar Top Selector

- date: `2026-04-27`
- status: `top_sidecar_selector_fails_smoke`
- git commit: `5d3c2d4103ca2ec9a9eca1cdb4cd74163cedecb6`
- min confidence: `0.01`

| Condition | Correct | Accepted | Clean Correct | Accepted Harm |
|---|---:|---:|---:|---:|
| `matched` | 0/6 | 2 | 0 | 0 |
| `shuffled_source` | 0/6 | 0 | 0 | 0 |
| `random_sidecar` | 1/6 | 3 | 1 | 0 |
| `target_only` | 0/6 | 0 | 0 | 0 |
| `slots_only` | 0/6 | 0 | 0 | 0 |

- source-necessary clean IDs: none
- control clean union IDs: `575d7e83d84c1e67`

## Command

```bash
scripts/analyze_candidate_score_sidecar_top_select.py --target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json --sidecar-jsonl results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars_predonly_no_t2t/live_candidate_sidecars.jsonl --min-confidence 0.01 --min-source-necessary-clean 1 --max-control-clean-union 0 --max-accepted-harm 0 --date 2026-04-27 --output-json results/svamp32_process_trace_sidecar_clean6_20260427/top_selector_process_trace_predonly_no_t2t.json --output-md results/svamp32_process_trace_sidecar_clean6_20260427/top_selector_process_trace_predonly_no_t2t.md
```
