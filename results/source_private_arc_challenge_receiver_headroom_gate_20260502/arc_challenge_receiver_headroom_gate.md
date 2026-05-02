# Source-Private ARC-Challenge Receiver/Headroom Gate

- date: `2026-05-02`
- pass gate: `False`
- receiver candidate pass: `False`
- strict per-seed CI pass count: `0 / 5`
- aggregate seed-row CI vs packet: `{'mean': -0.01416382252559727, 'ci95_low': -0.021843003412969283, 'ci95_high': -0.006821672354948829}`
- train / validation / test rows: `1119` / `299` / `1172`
- source packet budget: `12B`

## Default Seed Test

- matched packet-only accuracy: `0.344`
- public target receiver accuracy: `0.277`
- packet+target receiver accuracy: `0.340`
- receiver minus packet: `-0.004`
- paired CI95 vs packet: `{'mean': -0.004266211604095563, 'ci95_low': -0.00938566552901024, 'ci95_high': 0.0008532423208191126}`
- receiver minus public target: `0.062`

## Test Conditions

| Seed | Condition | Base | Receiver | Delta | CI95 low | Overrides |
|---:|---|---:|---:|---:|---:|---:|
| 47 | `matched_source_private_packet` | 0.344 | 0.340 | -0.004 | -0.009 | 156 |
| 47 | `source_label_copy` | 0.346 | 0.277 | -0.068 | -0.105 | 1172 |
| 47 | `same_byte_structured_text` | 0.311 | 0.311 | 0.000 | 0.000 | 0 |
| 47 | `target_public_ridge` | 0.277 | 0.277 | 0.000 | 0.000 | 1172 |
| 47 | `zero_source` | 0.265 | 0.277 | 0.012 | -0.024 | 1172 |
| 47 | `shuffled_source_packet` | 0.265 | 0.277 | 0.012 | -0.023 | 1172 |
| 47 | `random_same_byte_packet` | 0.243 | 0.277 | 0.034 | -0.002 | 1172 |
| 47 | `target_derived_sidecar` | 0.265 | 0.265 | -0.001 | -0.007 | 146 |
| 47 | `candidate_derangement` | 0.215 | 0.217 | 0.002 | -0.002 | 153 |
| 47 | `label_permutation` | 0.344 | 0.340 | -0.004 | -0.010 | 156 |
| 53 | `matched_source_private_packet` | 0.344 | 0.340 | -0.004 | -0.013 | 265 |
| 53 | `source_label_copy` | 0.346 | 0.346 | 0.000 | 0.000 | 0 |
| 53 | `same_byte_structured_text` | 0.311 | 0.255 | -0.055 | -0.088 | 868 |
| 53 | `target_public_ridge` | 0.277 | 0.277 | 0.000 | 0.000 | 0 |
| 53 | `zero_source` | 0.265 | 0.265 | 0.000 | 0.000 | 0 |
| 53 | `shuffled_source_packet` | 0.252 | 0.252 | 0.000 | 0.000 | 1 |
| 53 | `random_same_byte_packet` | 0.247 | 0.247 | 0.000 | 0.000 | 0 |
| 53 | `target_derived_sidecar` | 0.265 | 0.265 | 0.000 | -0.010 | 254 |
| 53 | `candidate_derangement` | 0.215 | 0.216 | 0.001 | -0.008 | 279 |
| 53 | `label_permutation` | 0.344 | 0.340 | -0.004 | -0.014 | 265 |
| 59 | `matched_source_private_packet` | 0.343 | 0.323 | -0.020 | -0.032 | 349 |
| 59 | `source_label_copy` | 0.346 | 0.346 | 0.000 | 0.000 | 0 |
| 59 | `same_byte_structured_text` | 0.311 | 0.256 | -0.055 | -0.089 | 962 |
| 59 | `target_public_ridge` | 0.277 | 0.277 | 0.000 | 0.000 | 0 |
| 59 | `zero_source` | 0.265 | 0.265 | 0.000 | 0.000 | 0 |
| 59 | `shuffled_source_packet` | 0.240 | 0.240 | 0.000 | 0.000 | 1 |
| 59 | `random_same_byte_packet` | 0.250 | 0.250 | 0.000 | 0.000 | 0 |
| 59 | `target_derived_sidecar` | 0.265 | 0.265 | 0.001 | -0.009 | 300 |
| 59 | `candidate_derangement` | 0.216 | 0.219 | 0.003 | -0.005 | 322 |
| 59 | `label_permutation` | 0.343 | 0.323 | -0.020 | -0.032 | 349 |
| 61 | `matched_source_private_packet` | 0.344 | 0.332 | -0.012 | -0.023 | 392 |
| 61 | `source_label_copy` | 0.346 | 0.346 | 0.000 | 0.000 | 283 |
| 61 | `same_byte_structured_text` | 0.311 | 0.256 | -0.055 | -0.089 | 1016 |
| 61 | `target_public_ridge` | 0.277 | 0.277 | 0.000 | 0.000 | 1096 |
| 61 | `zero_source` | 0.265 | 0.265 | 0.000 | 0.000 | 254 |
| 61 | `shuffled_source_packet` | 0.279 | 0.279 | 0.000 | 0.000 | 276 |
| 61 | `random_same_byte_packet` | 0.253 | 0.253 | 0.000 | 0.000 | 293 |
| 61 | `target_derived_sidecar` | 0.265 | 0.266 | 0.001 | -0.008 | 352 |
| 61 | `candidate_derangement` | 0.215 | 0.216 | 0.001 | -0.010 | 380 |
| 61 | `label_permutation` | 0.344 | 0.332 | -0.012 | -0.024 | 392 |
| 67 | `matched_source_private_packet` | 0.344 | 0.313 | -0.031 | -0.047 | 490 |
| 67 | `source_label_copy` | 0.346 | 0.346 | 0.000 | 0.000 | 305 |
| 67 | `same_byte_structured_text` | 0.311 | 0.256 | -0.055 | -0.087 | 1018 |
| 67 | `target_public_ridge` | 0.277 | 0.277 | 0.000 | 0.000 | 1126 |
| 67 | `zero_source` | 0.265 | 0.265 | 0.000 | 0.000 | 275 |
| 67 | `shuffled_source_packet` | 0.235 | 0.235 | 0.000 | 0.000 | 261 |
| 67 | `random_same_byte_packet` | 0.263 | 0.263 | 0.000 | 0.000 | 286 |
| 67 | `target_derived_sidecar` | 0.265 | 0.269 | 0.003 | -0.011 | 457 |
| 67 | `candidate_derangement` | 0.215 | 0.212 | -0.003 | -0.017 | 468 |
| 67 | `label_permutation` | 0.344 | 0.313 | -0.031 | -0.048 | 490 |

## Interpretation

ARC-Challenge tests whether the OpenBookQA receiver-fusion method generalizes back to the primary benchmark surface. A pass would upgrade the method from an OpenBookQA-only positive row to a cross-science-QA evidence-fusion branch; a fail would demote receiver-fusion as benchmark-specific calibration and make common-basis or hidden-innovation compression the next live branch.

Lay description: the experiment asks whether the receiver can learn when to trust the tiny source packet and when to fall back to its own public question/candidate scorer. The source packet is like a short hint; the receiver is a trained referee that decides whether the hint looks useful for this question.
