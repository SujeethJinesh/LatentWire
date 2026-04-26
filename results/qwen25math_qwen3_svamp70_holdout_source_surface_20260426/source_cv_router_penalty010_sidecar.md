# Source Sidecar CV Router Gate

- date: `2026-04-26`
- status: `source_sidecar_cv_router_fails_gate`
- reference rows: `70`
- outer folds: `5`
- features: `source_prediction_char_count, source_target_len_ratio, source_numeric_count, source_generated_tokens, source_has_final_marker`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Clean Necessary | Control Clean Union | Accepted Harm | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---|---|
| 2,3 | 1 | source_sidecar_cv_router_fails_gate | 6 | 0 | 0 | 2 | none | min_correct, min_clean_source_necessary |
| 2,3,5 | 1 | source_sidecar_cv_router_fails_gate | 6 | 0 | 0 | 2 | none | min_correct, min_clean_source_necessary |
| 2,3,5,7 | 1 | source_sidecar_cv_router_fails_gate | 6 | 0 | 0 | 2 | none | min_correct, min_clean_source_necessary |
| 97 | 1 | source_sidecar_cv_router_fails_gate | 6 | 0 | 0 | 2 | none | min_correct, min_clean_source_necessary |

## Fold Rules

### Moduli 2,3

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 1 | `source_generated_tokens` | `le` | 56.0 | 1 | 0 | 4 | 0.6000 |
| 2 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 3 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 4 | `source_prediction_char_count` | `le` | 247.0 | 4 | 0 | 29 | 1.1000 |

### Moduli 2,3,5

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 1 | `source_has_final_marker` | `ge` | 0.5 | 5 | 2 | 21 | 0.9000 |
| 2 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 3 | `source_target_len_ratio` | `le` | 1.065217391304348 | 5 | 1 | 26 | 1.4000 |
| 4 | `source_has_final_marker` | `ge` | 0.5 | 5 | 0 | 20 | 3.0000 |

### Moduli 2,3,5,7

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 1 | `source_has_final_marker` | `ge` | 0.5 | 5 | 2 | 21 | 0.9000 |
| 2 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 3 | `source_target_len_ratio` | `le` | 1.065217391304348 | 5 | 1 | 26 | 1.4000 |
| 4 | `source_has_final_marker` | `ge` | 0.5 | 5 | 0 | 20 | 3.0000 |

### Moduli 97

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 1 | `source_has_final_marker` | `ge` | 0.5 | 5 | 2 | 21 | 0.9000 |
| 2 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 2 | 0.8000 |
| 3 | `source_target_len_ratio` | `le` | 1.065217391304348 | 5 | 1 | 26 | 1.4000 |
| 4 | `source_has_final_marker` | `ge` | 0.5 | 5 | 0 | 20 | 3.0000 |
