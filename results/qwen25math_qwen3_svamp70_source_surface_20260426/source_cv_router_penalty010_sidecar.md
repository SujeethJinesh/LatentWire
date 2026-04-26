# Source Sidecar CV Router Gate

- date: `2026-04-26`
- status: `source_sidecar_cv_router_clears_gate`
- reference rows: `70`
- outer folds: `5`
- features: `source_prediction_char_count, source_target_len_ratio, source_numeric_count, source_generated_tokens, source_has_final_marker`

## Moduli Sweep

| Moduli | Bytes | Status | Matched | Clean Necessary | Control Clean Union | Accepted Harm | Source-Necessary IDs | Failing Criteria |
|---|---:|---|---:|---:|---:|---:|---|---|
| 2,3 | 1 | source_sidecar_cv_router_fails_gate | 20 | 0 | 0 | 1 | none | min_correct, min_clean_source_necessary |
| 2,3,5 | 1 | source_sidecar_cv_router_fails_gate | 22 | 2 | 0 | 1 | `4d780f825bb8541c`, `ce08a3a269bf0151` | min_correct, min_clean_source_necessary |
| 2,3,5,7 | 1 | source_sidecar_cv_router_clears_gate | 25 | 4 | 0 | 1 | `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`, `ce08a3a269bf0151` | none |
| 97 | 1 | source_sidecar_cv_router_clears_gate | 25 | 4 | 0 | 1 | `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`, `ce08a3a269bf0151` | none |

## Fold Rules

### Moduli 2,3

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_target_len_ratio` | `le` | 0.5307692307692308 | 0 | 0 | 1 | -0.1000 |
| 1 | `source_target_len_ratio` | `le` | 0.5307692307692308 | 0 | 0 | 1 | -0.1000 |
| 2 | `source_target_len_ratio` | `le` | 0.5307692307692308 | 0 | 0 | 1 | -0.1000 |
| 3 | `source_target_len_ratio` | `le` | 0.9777777777777777 | 2 | 0 | 13 | 0.7000 |
| 4 | `source_target_len_ratio` | `le` | 0.9777777777777777 | 2 | 1 | 10 | 0.0000 |

### Moduli 2,3,5

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 5 | 1 | 17 | 2.3000 |
| 1 | `source_target_len_ratio` | `le` | 0.9777777777777777 | 4 | 1 | 14 | 1.6000 |
| 2 | `source_numeric_count` | `le` | 1.0 | 1 | 0 | 1 | 0.9000 |
| 3 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 4 | 0 | 14 | 2.6000 |
| 4 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 4 | 1 | 11 | 1.9000 |

### Moduli 2,3,5,7

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 6 | 1 | 17 | 3.3000 |
| 1 | `source_target_len_ratio` | `le` | 0.9777777777777777 | 5 | 1 | 14 | 2.6000 |
| 2 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 4 | 1 | 12 | 1.8000 |
| 3 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 5 | 0 | 14 | 3.6000 |
| 4 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 4 | 1 | 11 | 1.9000 |

### Moduli 97

| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 6 | 1 | 17 | 3.3000 |
| 1 | `source_target_len_ratio` | `le` | 0.9777777777777777 | 5 | 1 | 14 | 2.6000 |
| 2 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 4 | 1 | 12 | 1.8000 |
| 3 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 5 | 0 | 14 | 3.6000 |
| 4 | `source_target_len_ratio` | `le` | 0.9911111111111112 | 4 | 1 | 11 | 1.9000 |
