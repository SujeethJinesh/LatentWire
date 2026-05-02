# ARC/OpenBookQA Soft-Prefix Preflight

- date: `2026-05-02`
- benchmark: `ARC-Challenge`
- pass gate: `False`
- fit/eval rows: `4` / `4`
- matched accuracy: `0.250`
- best control by accuracy: `target_only`
- best control accuracy: `0.250`
- matched margin: `-0.551407`
- best control margin: `-0.317793`
- matched minus best-control margin: `-0.233614`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_soft_prefix` | 0.250 | 1 / 4 | -0.551407 |
| `target_only` | 0.250 | 1 / 4 | -0.391752 |
| `target_cache_only_prefix` | 0.250 | 1 / 4 | -0.651156 |
| `slots_only_prefix` | 0.250 | 1 / 4 | -0.317793 |
| `zero_source` | 0.250 | 1 / 4 | -0.746072 |
| `shuffled_source` | 0.000 | 0 / 4 | -1.734032 |
| `same_norm_noise` | 0.250 | 1 / 4 | -0.424517 |
| `train_mean_source` | 0.250 | 1 / 4 | -1.309134 |
| `label_shuffled` | 0.000 | 0 / 4 | -2.657316 |
| `candidate_roll_source` | 0.250 | 1 / 4 | -0.551421 |
| `candidate_derangement` | 0.250 | 1 / 4 | -1.002580 |
| `same_byte_visible_text` | 0.250 | 1 / 4 | -0.432369 |
| `source_label_copy_audit_upper_bound` | 0.750 | 3 / 4 | 500000000.000000 |

## Interpretation

This preflight passes only if the matched soft-prefix uses source information that the target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak.

Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden source-model summary into soft tokens prepended to the target model. The controls ask whether the soft tokens are really using the source row, or whether a static/target-only prefix can do the same thing.
