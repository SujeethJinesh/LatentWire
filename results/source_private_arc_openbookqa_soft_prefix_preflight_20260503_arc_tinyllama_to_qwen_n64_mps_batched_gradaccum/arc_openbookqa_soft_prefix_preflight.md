# ARC/OpenBookQA Soft-Prefix Preflight

- date: `2026-05-03`
- benchmark: `ARC-Challenge`
- pass gate: `False`
- fit/eval rows: `32` / `32`
- matched accuracy: `0.219`
- best control by accuracy: `same_byte_visible_text`
- best control accuracy: `0.500`
- matched margin: `-0.718780`
- best control margin: `0.021598`
- matched minus best-control margin: `-0.740378`

## Conditions

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| `matched_soft_prefix` | 0.219 | 7 / 32 | -0.718780 |
| `target_only` | 0.406 | 13 / 32 | -0.097709 |
| `source_free_prefix` | 0.344 | 11 / 32 | -0.324237 |
| `target_cache_only_prefix` | 0.250 | 8 / 32 | -0.620005 |
| `slots_only_prefix` | 0.469 | 15 / 32 | -0.031533 |
| `zero_source` | 0.250 | 8 / 32 | -0.393371 |
| `shuffled_source` | 0.375 | 12 / 32 | -0.540985 |
| `same_norm_noise` | 0.281 | 9 / 32 | -0.252443 |
| `train_mean_source` | 0.375 | 12 / 32 | -0.452005 |
| `label_shuffled` | 0.219 | 7 / 32 | -0.522261 |
| `candidate_roll_source` | 0.219 | 7 / 32 | -0.718087 |
| `candidate_derangement` | 0.188 | 6 / 32 | -0.875112 |
| `same_byte_visible_text` | 0.500 | 16 / 32 | 0.021598 |
| `packet_only_source_index` | 0.469 | 15 / 32 | -0.062500 |
| `qwen_substituted_packet` | 0.438 | 14 / 32 | -0.125000 |
| `source_label_copy_audit_upper_bound` | 0.469 | 15 / 32 | -62500000.000000 |

## Interpretation

This preflight passes only if the matched soft-prefix uses source information that the target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak.

Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden source-model summary into soft tokens prepended to the target model. The controls ask whether the soft tokens are really using the source row, or whether a static/target-only prefix can do the same thing.
