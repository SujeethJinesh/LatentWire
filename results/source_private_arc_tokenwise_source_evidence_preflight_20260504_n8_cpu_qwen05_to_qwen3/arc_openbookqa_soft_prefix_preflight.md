# ARC/OpenBookQA Soft-Prefix Preflight

- date: `2026-05-04`
- benchmark: `ARC-Challenge`
- pass gate: `False`
- fit/eval rows: `4` / `4`
- matched accuracy: `0.000`
- best control by accuracy: `packet_only_source_index`
- best control accuracy: `0.750`
- matched margin: `-3.356703`
- best control margin: `1.241773`
- matched minus best-control margin: `-4.598476`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_soft_prefix` | 0.000 | 0 / 4 | -3.356703 |
| `target_only` | 0.250 | 1 / 4 | 0.431175 |
| `source_free_prefix` | 0.000 | 0 / 4 | -9.906060 |
| `target_cache_only_prefix` | 0.000 | 0 / 4 | -2.428298 |
| `target_derived_prefix` | 0.000 | 0 / 4 | -9.906060 |
| `slots_only_prefix` | 0.500 | 2 / 4 | -0.746434 |
| `zero_source` | 0.000 | 0 / 4 | -6.052378 |
| `shuffled_source` | 0.250 | 1 / 4 | -3.447245 |
| `source_row_shuffle` | 0.250 | 1 / 4 | -3.447245 |
| `same_norm_noise` | 0.000 | 0 / 4 | -4.212645 |
| `train_mean_source` | 0.000 | 0 / 4 | -2.239609 |
| `atom_shuffle` | 0.500 | 2 / 4 | -0.710697 |
| `coefficient_shuffle` | 0.000 | 0 / 4 | -3.346678 |
| `top_atom_knockout` | 0.000 | 0 / 4 | -3.010803 |
| `label_shuffled` | 0.000 | 0 / 4 | -4.503169 |
| `candidate_roll_source` | 0.000 | 0 / 4 | -3.356766 |
| `candidate_roll` | 0.000 | 0 / 4 | -3.356766 |
| `candidate_score_roll_source` | 0.000 | 0 / 4 | -3.731837 |
| `candidate_derangement` | 0.250 | 1 / 4 | -2.348884 |
| `same_byte_visible_text` | 0.500 | 2 / 4 | 1.241773 |
| `packet_only_source_index` | 0.750 | 3 / 4 | 0.500000 |
| `source_rank_control` | 0.750 | 3 / 4 | 0.000000 |
| `source_score_control` | 0.750 | 3 / 4 | 1.154701 |
| `qwen_substituted_packet` | 0.750 | 3 / 4 | 0.500000 |
| `source_label_copy_audit_upper_bound` | 0.750 | 3 / 4 | 500000000.000000 |

## Interpretation

This preflight passes only if the matched soft-prefix uses source information that the target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak.

Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden source-model summary into soft tokens prepended to the target model. The controls ask whether the soft tokens are really using the source row, or whether a static/target-only prefix can do the same thing.
