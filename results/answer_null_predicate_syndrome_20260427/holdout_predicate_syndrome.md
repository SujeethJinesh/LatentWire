# Answer-Null Predicate Syndrome

- date: `2026-04-27`
- status: `answer_null_predicate_syndrome_fails_smoke`
- git commit: `ee675b86c24364dae673161a3420dba1b284163f`
- min confidence: `0.0`

| Condition | Correct | Accepted | Clean Correct | Accepted Harm |
|---|---:|---:|---:|---:|
| `matched` | 13/70 | 46 | 1 | 5 |
| `shuffled_source` | 12/70 | 45 | 1 | 5 |
| `random_syndrome` | 14/70 | 48 | 1 | 5 |
| `target_only` | 12/70 | 48 | 1 | 6 |
| `slots_only` | 12/70 | 48 | 1 | 6 |

- source-necessary clean IDs: none
- control clean union IDs: `ab1e71e8928661d0`

## Command

```bash
scripts/analyze_answer_null_predicate_syndrome.py --target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json --min-confidence 0.0 --min-source-necessary-clean 1 --max-control-clean-union 0 --max-accepted-harm 0 --date 2026-04-27 --output-json results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.json --output-md results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.md
```
