# Source-Private OpenBookQA Receiver/Headroom Gate

- date: `2026-05-02`
- pass gate: `True`
- receiver candidate pass: `True`
- strict per-seed CI pass count: `2 / 5`
- aggregate seed-row CI vs packet: `{'mean': 0.033600000000000005, 'ci95_low': 0.00040000000000000034, 'ci95_high': 0.06560999999999995}`
- train / validation / test rows: `4957` / `500` / `500`
- source packet budget: `3B`

## Default Seed Test

- matched packet-only accuracy: `0.378`
- public target receiver accuracy: `0.372`
- packet+target receiver accuracy: `0.424`
- receiver minus packet: `0.046`
- paired CI95 vs packet: `{'mean': 0.046, 'ci95_low': 0.007950000000000002, 'ci95_high': 0.08404999999999974}`
- receiver minus public target: `0.052`

## Test Conditions

| Seed | Condition | Base | Receiver | Delta | CI95 low | Overrides |
|---:|---|---:|---:|---:|---:|---:|
| 47 | `matched_source_private_packet` | 0.378 | 0.424 | 0.046 | 0.008 | 140 |
| 47 | `source_label_copy` | 0.378 | 0.372 | -0.006 | -0.066 | 480 |
| 47 | `same_byte_structured_text` | 0.350 | 0.378 | 0.028 | 0.000 | 98 |
| 47 | `target_public_ridge` | 0.372 | 0.372 | 0.000 | 0.000 | 418 |
| 47 | `zero_source` | 0.276 | 0.372 | 0.096 | 0.040 | 475 |
| 47 | `shuffled_source_packet` | 0.250 | 0.364 | 0.114 | 0.060 | 400 |
| 47 | `random_same_byte_packet` | 0.228 | 0.370 | 0.142 | 0.088 | 421 |
| 47 | `target_derived_sidecar` | 0.276 | 0.352 | 0.076 | 0.040 | 156 |
| 47 | `candidate_derangement` | 0.210 | 0.292 | 0.082 | 0.046 | 145 |
| 47 | `label_permutation` | 0.378 | 0.424 | 0.046 | 0.008 | 140 |
| 53 | `matched_source_private_packet` | 0.378 | 0.418 | 0.040 | -0.008 | 273 |
| 53 | `source_label_copy` | 0.378 | 0.414 | 0.036 | -0.014 | 299 |
| 53 | `same_byte_structured_text` | 0.350 | 0.430 | 0.080 | 0.036 | 271 |
| 53 | `target_public_ridge` | 0.372 | 0.372 | 0.000 | 0.000 | 34 |
| 53 | `zero_source` | 0.276 | 0.352 | 0.076 | 0.028 | 313 |
| 53 | `shuffled_source_packet` | 0.270 | 0.374 | 0.104 | 0.052 | 323 |
| 53 | `random_same_byte_packet` | 0.234 | 0.362 | 0.128 | 0.080 | 334 |
| 53 | `target_derived_sidecar` | 0.274 | 0.364 | 0.090 | 0.044 | 303 |
| 53 | `candidate_derangement` | 0.212 | 0.346 | 0.134 | 0.090 | 291 |
| 53 | `label_permutation` | 0.378 | 0.418 | 0.040 | -0.008 | 273 |
| 59 | `matched_source_private_packet` | 0.378 | 0.422 | 0.044 | 0.012 | 102 |
| 59 | `source_label_copy` | 0.378 | 0.400 | 0.022 | 0.004 | 35 |
| 59 | `same_byte_structured_text` | 0.350 | 0.372 | 0.022 | -0.034 | 500 |
| 59 | `target_public_ridge` | 0.372 | 0.372 | 0.000 | 0.000 | 0 |
| 59 | `zero_source` | 0.276 | 0.304 | 0.028 | 0.012 | 37 |
| 59 | `shuffled_source_packet` | 0.270 | 0.316 | 0.046 | 0.024 | 72 |
| 59 | `random_same_byte_packet` | 0.244 | 0.288 | 0.044 | 0.018 | 78 |
| 59 | `target_derived_sidecar` | 0.276 | 0.316 | 0.040 | 0.012 | 108 |
| 59 | `candidate_derangement` | 0.210 | 0.270 | 0.060 | 0.030 | 96 |
| 59 | `label_permutation` | 0.378 | 0.422 | 0.044 | 0.014 | 102 |
| 61 | `matched_source_private_packet` | 0.380 | 0.394 | 0.014 | -0.036 | 308 |
| 61 | `source_label_copy` | 0.378 | 0.372 | -0.006 | -0.066 | 500 |
| 61 | `same_byte_structured_text` | 0.350 | 0.350 | 0.000 | 0.000 | 0 |
| 61 | `target_public_ridge` | 0.372 | 0.372 | 0.000 | 0.000 | 500 |
| 61 | `zero_source` | 0.276 | 0.372 | 0.096 | 0.038 | 500 |
| 61 | `shuffled_source_packet` | 0.264 | 0.372 | 0.108 | 0.050 | 493 |
| 61 | `random_same_byte_packet` | 0.256 | 0.370 | 0.114 | 0.058 | 497 |
| 61 | `target_derived_sidecar` | 0.276 | 0.364 | 0.088 | 0.042 | 308 |
| 61 | `candidate_derangement` | 0.208 | 0.332 | 0.124 | 0.078 | 317 |
| 61 | `label_permutation` | 0.380 | 0.394 | 0.014 | -0.038 | 308 |
| 67 | `matched_source_private_packet` | 0.378 | 0.402 | 0.024 | -0.004 | 84 |
| 67 | `source_label_copy` | 0.378 | 0.376 | -0.002 | -0.006 | 3 |
| 67 | `same_byte_structured_text` | 0.350 | 0.372 | 0.022 | -0.036 | 500 |
| 67 | `target_public_ridge` | 0.372 | 0.372 | 0.000 | 0.000 | 0 |
| 67 | `zero_source` | 0.276 | 0.276 | 0.000 | -0.006 | 3 |
| 67 | `shuffled_source_packet` | 0.278 | 0.292 | 0.014 | 0.004 | 12 |
| 67 | `random_same_byte_packet` | 0.250 | 0.262 | 0.012 | 0.004 | 10 |
| 67 | `target_derived_sidecar` | 0.274 | 0.316 | 0.042 | 0.020 | 74 |
| 67 | `candidate_derangement` | 0.210 | 0.248 | 0.038 | 0.010 | 80 |
| 67 | `label_permutation` | 0.378 | 0.402 | 0.024 | -0.004 | 84 |

## Interpretation

OpenBookQA now has a positive receiver-fusion row: the default 3B packet receiver improves over packet-only and over a train-split public target scorer on held-out test, while same-byte text and source-destroy controls stay lower. The result is a useful positive method branch, but it should be framed as source-private evidence fusion rather than universal latent-language transfer; the current packet still behaves like a compact source-selected-candidate sketch, so a stronger common-basis or learned connector remains necessary for a comfortable ICLR full paper.

Lay description: the experiment asks whether the receiver can learn when to trust the tiny source packet and when to fall back to its own public question/candidate scorer. The source packet is like a short hint; the receiver is a trained referee that decides whether the hint looks useful for this question.
