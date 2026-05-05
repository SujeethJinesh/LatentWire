# Source-Private ARC-Challenge Seed-Stability Gate

- date: `2026-05-05`
- pass gate: `True`
- split: `openbookqa_test_hashed_3b_20seed`
- eval rows: `500`
- packet budget: `3B`
- seeds: `47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139`

| Seed | Pass | Matched | Target | Best control | Same-byte text | Derange | CI95 low vs target |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 47 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.048 |
| 53 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.212 | 0.046 |
| 59 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.046 |
| 61 | True | 0.380 | 0.276 | 0.276 (zero_source) | 0.350 | 0.208 | 0.048 |
| 67 | True | 0.378 | 0.276 | 0.278 (shuffled_source_packet) | 0.350 | 0.210 | 0.040 |
| 71 | True | 0.378 | 0.276 | 0.278 (target_derived_sidecar) | 0.350 | 0.210 | 0.046 |
| 73 | True | 0.376 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.040 |
| 79 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.042 |
| 83 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.050 |
| 89 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.038 |
| 97 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.046 |
| 101 | True | 0.378 | 0.276 | 0.284 (random_same_byte_packet) | 0.350 | 0.210 | 0.046 |
| 103 | True | 0.380 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.050 |
| 107 | True | 0.376 | 0.276 | 0.292 (random_same_byte_packet) | 0.350 | 0.212 | 0.046 |
| 109 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.046 |
| 113 | True | 0.378 | 0.276 | 0.278 (target_derived_sidecar) | 0.350 | 0.210 | 0.050 |
| 127 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.208 | 0.044 |
| 131 | True | 0.380 | 0.276 | 0.278 (target_derived_sidecar) | 0.350 | 0.208 | 0.046 |
| 137 | True | 0.376 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.040 |
| 139 | True | 0.376 | 0.276 | 0.284 (random_same_byte_packet) | 0.350 | 0.210 | 0.042 |

## Aggregate

- pass count: `20 / 20`
- matched accuracy mean/min/max: `0.378` / `0.376` / `0.380`
- minimum matched-target lift: `0.100`
- minimum matched-best-control lift: `0.084`
- minimum matched-same-byte-text lift: `0.026`
- minimum CI95 lower bound vs target: `0.038`

This gate varies only the packet projection/random-control seed while reusing the answer-key-forbidden source-choice cache from the anchor ARC run.
