# Source-Private ARC-Challenge Seed-Stability Gate

- date: `2026-05-01`
- pass gate: `True`
- split: `openbookqa_test_hashed_3b`
- eval rows: `500`
- packet budget: `3B`
- seeds: `47, 53, 59, 61, 67`

| Seed | Pass | Matched | Target | Best control | Same-byte text | Derange | CI95 low vs target |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 47 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.046 |
| 53 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.212 | 0.041 |
| 59 | True | 0.378 | 0.276 | 0.276 (zero_source) | 0.350 | 0.210 | 0.043 |
| 61 | True | 0.380 | 0.276 | 0.276 (zero_source) | 0.350 | 0.208 | 0.048 |
| 67 | True | 0.378 | 0.276 | 0.278 (shuffled_source_packet) | 0.350 | 0.210 | 0.038 |

## Aggregate

- pass count: `5 / 5`
- matched accuracy mean/min/max: `0.378` / `0.378` / `0.380`
- minimum matched-target lift: `0.102`
- minimum matched-best-control lift: `0.100`
- minimum matched-same-byte-text lift: `0.028`
- minimum CI95 lower bound vs target: `0.038`

This gate varies only the packet projection/random-control seed while reusing the answer-key-forbidden source-choice cache from the anchor ARC run.
