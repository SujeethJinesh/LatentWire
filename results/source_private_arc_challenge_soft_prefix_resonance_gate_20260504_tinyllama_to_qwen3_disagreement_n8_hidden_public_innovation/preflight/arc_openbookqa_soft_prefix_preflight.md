# ARC/OpenBookQA Soft-Prefix Preflight

- date: `2026-05-04`
- benchmark: `ARC-Challenge-disagreement`
- pass gate: `False`
- fit/eval rows: `8` / `8`
- matched accuracy: `0.500`
- best control by accuracy: `qwen_substituted_packet`
- best control accuracy: `0.625`
- matched margin: `-1.501413`
- best control margin: `1.357327`
- matched minus best-control margin: `-2.858740`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_soft_prefix` | 0.500 | 4 / 8 | -1.501413 |
| `target_only` | 0.500 | 4 / 8 | 1.357327 |
| `source_free_prefix` | 0.500 | 4 / 8 | 0.538085 |
| `target_cache_only_prefix` | 0.500 | 4 / 8 | 0.353054 |
| `target_derived_prefix` | 0.500 | 4 / 8 | 0.538085 |
| `slots_only_prefix` | 0.500 | 4 / 8 | 1.321455 |
| `zero_source` | 0.250 | 2 / 8 | -3.004782 |
| `shuffled_source` | 0.250 | 2 / 8 | -1.942923 |
| `source_row_shuffle` | 0.250 | 2 / 8 | -1.942923 |
| `same_norm_noise` | 0.250 | 2 / 8 | -3.207169 |
| `train_mean_source` | 0.125 | 1 / 8 | -3.247572 |
| `label_shuffled` | 0.500 | 4 / 8 | 0.210773 |
| `candidate_roll_source` | 0.500 | 4 / 8 | -1.501417 |
| `candidate_roll` | 0.500 | 4 / 8 | -1.501417 |
| `candidate_score_roll_source` | 0.500 | 4 / 8 | -0.993314 |
| `candidate_derangement` | 0.250 | 2 / 8 | -1.265035 |
| `same_byte_visible_text` | 0.500 | 4 / 8 | 1.282678 |
| `packet_only_source_index` | 0.125 | 1 / 8 | -0.750000 |
| `source_rank_control` | 0.125 | 1 / 8 | -0.343750 |
| `source_score_control` | 0.125 | 1 / 8 | -0.994205 |
| `qwen_substituted_packet` | 0.625 | 5 / 8 | 0.250000 |
| `source_label_copy_audit_upper_bound` | 0.125 | 1 / 8 | -750000000.000000 |

## Interpretation

This preflight passes only if the matched soft-prefix uses source information that the target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak.

Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden source-model summary into soft tokens prepended to the target model. The controls ask whether the soft tokens are really using the source row, or whether a static/target-only prefix can do the same thing.
