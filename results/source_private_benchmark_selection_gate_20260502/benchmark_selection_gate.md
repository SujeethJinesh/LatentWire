# Source-Private Benchmark Selection Gate

- pass gate: `True`
- recommended next gate: `train-only receiver/headroom gate on OpenBookQA test using the promoted 3B packet, target-cache-only, candidate-only, target-derived, row-shuffle, random same-rate, label-permutation, candidate-derangement, same-byte text, and source-label-copy controls`
- selected benchmark: `OpenBookQA test`
- receiver candidates: `2`
- text-saturated diagnostics: `1`

## Rows

| Benchmark | Budget | Seeds | Matched | Target | Text | Lift vs target | Lift vs text | Oracle headroom | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenBookQA `test` | 3B | 5/5 | 0.378 | 0.276 | 0.350 | 0.102 | 0.028 | 0.164 | `receiver_candidate` |
| ARC-Challenge `test` | 12B | 5/5 | 0.344 | 0.265 | 0.311 | 0.078 | 0.032 | 0.185 | `receiver_candidate` |
| CommonsenseQA `validation` | 2B | 0/5 | 0.438 | 0.206 | 0.424 | 0.232 | 0.013 | 0.118 | `diagnostic_text_saturated` |

## Interpretation

OpenBookQA and ARC both retain large target-or-packet oracle headroom after fixed-packet seed stability, while CommonsenseQA remains useful but text-saturated. This makes OpenBookQA the highest-value next ICLR method gate: it is already a second public benchmark, has a strict text-margin packet row, and has enough target/packet complementarity to test a real receiver rather than another HellaSwag hidden-code variant.

## Next Gate

train-only receiver/headroom gate on OpenBookQA test using the promoted 3B packet, target-cache-only, candidate-only, target-derived, row-shuffle, random same-rate, label-permutation, candidate-derangement, same-byte text, and source-label-copy controls
