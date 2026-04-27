# Candidate-Score Sidecar Top Selector

- date: `2026-04-27`
- status: `top_sidecar_selector_fails_smoke`
- git commit: `ef4886ed050f7ee21c8746a42e673a86c5fc1fe1`
- min confidence: `0.0`

| Condition | Correct | Accepted | Clean Correct | Accepted Harm |
|---|---:|---:|---:|---:|
| `matched` | 6/32 | 17 | 0 | 5 |
| `zero_source` | 8/32 | 0 | 0 | 0 |
| `shuffled_source` | 6/32 | 11 | 0 | 3 |
| `random_sidecar` | 3/32 | 28 | 0 | 7 |
| `target_only` | 8/32 | 0 | 0 | 0 |
| `slots_only` | 8/32 | 0 | 0 | 0 |

- source-necessary clean IDs: none
- control clean union IDs: none

## Command

```bash
scripts/analyze_candidate_score_sidecar_top_select.py --target-set results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json --sidecar-jsonl results/svamp32_source_sample_selector_newclean2_20260427/sidecars_answer_only/live_candidate_sidecars.jsonl --min-confidence 0.0 --min-source-necessary-clean 1 --max-control-clean-union 0 --max-accepted-harm 0 --date 2026-04-27 --output-json results/svamp32_source_sample_selector_newclean2_20260427/top_select_answer_only.json --output-md results/svamp32_source_sample_selector_newclean2_20260427/top_select_answer_only.md
```
