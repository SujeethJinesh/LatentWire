# ARC/OpenBookQA Soft-Prefix Preflight

- date: `2026-05-02`
- benchmark: `ARC-Challenge`
- pass gate: `False`
- fit/eval rows: `4` / `4`
- matched accuracy: `0.250`
- best control by accuracy: `shuffled_source`
- best control accuracy: `0.500`
- matched margin: `-1.229876`
- best control margin: `-0.330309`
- matched minus best-control margin: `-0.899567`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_soft_prefix` | 0.250 | 1 / 4 | -1.229876 |
| `target_only` | 0.250 | 1 / 4 | -0.391752 |
| `target_cache_only_prefix` | 0.250 | 1 / 4 | -0.665684 |
| `slots_only_prefix` | 0.250 | 1 / 4 | -0.354785 |
| `zero_source` | 0.250 | 1 / 4 | -0.991535 |
| `shuffled_source` | 0.500 | 2 / 4 | -0.330309 |
| `same_norm_noise` | 0.250 | 1 / 4 | -0.899848 |
| `train_mean_source` | 0.250 | 1 / 4 | -1.437397 |
| `label_shuffled` | 0.000 | 0 / 4 | -0.828966 |
| `candidate_derangement` | 0.000 | 0 / 4 | -2.015325 |
| `same_byte_visible_text` | 0.250 | 1 / 4 | -0.432369 |
| `source_label_copy_audit_upper_bound` | 0.750 | 3 / 4 | 500000000.000000 |

## Interpretation

This preflight passes only if the matched soft-prefix uses source information that the target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak.

Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden source-model summary into soft tokens prepended to the target model. The controls ask whether the soft tokens are really using the source row, or whether a static/target-only prefix can do the same thing.
