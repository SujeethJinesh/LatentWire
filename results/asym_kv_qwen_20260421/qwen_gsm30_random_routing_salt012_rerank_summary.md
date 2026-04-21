# Stochastic Route Reranker Summary

| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| target_alone | 0.0667 | +0.0000 | 0 | 0 | 0 | 0 |
| rerank_agreement_then_format | 0.0667 | +0.0000 | 1 | 1 | 1 | 27 |
| rerank_format_then_agreement | 0.1333 | +0.0667 | 3 | 1 | 1 | 25 |
| rerank_seed_format_confidence | 0.1000 | +0.0333 | 3 | 2 | 0 | 25 |
| rerank_target_on_low_format | 0.1333 | +0.0667 | 3 | 1 | 1 | 25 |
| rerank_target_on_strict_format | 0.1667 | +0.1000 | 3 | 0 | 2 | 25 |
| rerank_agreement_or_target | 0.0333 | -0.0333 | 0 | 1 | 1 | 28 |
| rerank_numeric_consistency_then_completion | 0.0667 | +0.0000 | 2 | 2 | 0 | 26 |
| rerank_completion_then_numeric_consistency | 0.0667 | +0.0000 | 2 | 2 | 0 | 26 |
| rerank_numeric_consistency_or_target | 0.0667 | +0.0000 | 2 | 2 | 0 | 26 |

Interpretation:

Format-first reranking is the first non-oracle selector to test whether stochastic route candidates can be used without label leakage. Compare it to target-alone and to the oracle aggregate to separate candidate quality from selection quality.

The stricter target fallback is the current best non-oracle selector:
`rerank_target_on_strict_format` reaches `0.1667`, a `+0.1000` absolute gain
over target-alone with zero baseline-only losses. This is still well below the
target-or-seed oracle `0.3000`, so the next blocker is stronger verifier
selection rather than candidate generation.

Numeric ablation:

Numeric-consistency-first reranking is a disjoint ablation that uses only the candidate's own numeric text and completion cues. It checks whether the reranker can prefer self-consistent numeric answers even when format markers are weak or misleading.

On this GSM30 pool, numeric consistency by itself is diagnostic but not yet a
better selector: all three numeric policies land at `0.0667`.
