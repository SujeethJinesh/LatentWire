# ARC/OpenBookQA Soft-Prefix Preflight

- date: `2026-05-04`
- benchmark: `ARC-Challenge-disagreement`
- pass gate: `False`
- fit/eval rows: `8` / `8`
- matched accuracy: `0.250`
- best control by accuracy: `zero_source`
- best control accuracy: `0.750`
- matched margin: `-2.790504`
- best control margin: `1.501536`
- matched minus best-control margin: `-4.292040`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_soft_prefix` | 0.250 | 2 / 8 | -2.790504 |
| `target_only` | 0.500 | 4 / 8 | 1.357327 |
| `source_free_prefix` | 0.625 | 5 / 8 | -1.182415 |
| `target_cache_only_prefix` | 0.500 | 4 / 8 | 0.275707 |
| `target_derived_prefix` | 0.625 | 5 / 8 | -1.182415 |
| `slots_only_prefix` | 0.500 | 4 / 8 | 1.279127 |
| `zero_source` | 0.750 | 6 / 8 | 1.438470 |
| `shuffled_source` | 0.375 | 3 / 8 | -1.541090 |
| `source_row_shuffle` | 0.375 | 3 / 8 | -1.541090 |
| `same_norm_noise` | 0.375 | 3 / 8 | -2.215687 |
| `train_mean_source` | 0.375 | 3 / 8 | -2.236364 |
| `label_shuffled` | 0.250 | 2 / 8 | -3.850294 |
| `candidate_roll_source` | 0.250 | 2 / 8 | -2.790485 |
| `candidate_roll` | 0.250 | 2 / 8 | -2.790485 |
| `candidate_score_roll_source` | 0.375 | 3 / 8 | -2.880266 |
| `candidate_derangement` | 0.625 | 5 / 8 | 1.501536 |
| `same_byte_visible_text` | 0.500 | 4 / 8 | 1.282678 |
| `packet_only_source_index` | 0.125 | 1 / 8 | -0.750000 |
| `source_rank_control` | 0.125 | 1 / 8 | -0.343750 |
| `source_score_control` | 0.125 | 1 / 8 | -0.994205 |
| `qwen_substituted_packet` | 0.625 | 5 / 8 | 0.250000 |
| `source_label_copy_audit_upper_bound` | 0.125 | 1 / 8 | -750000000.000000 |

## Interpretation

This preflight passes only if the matched soft-prefix uses source information that the target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak.

Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden source-model summary into soft tokens prepended to the target model. The controls ask whether the soft tokens are really using the source row, or whether a static/target-only prefix can do the same thing.
