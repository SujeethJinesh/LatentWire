# Stochastic Route Reranker Summary

| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| target_alone | 0.0667 | +0.0000 | 0 | 0 | 0 | 0 |
| rerank_agreement_then_format | 0.0667 | +0.0000 | 1 | 1 | 1 | 27 |
| rerank_format_then_agreement | 0.1333 | +0.0667 | 3 | 1 | 1 | 25 |
| rerank_seed_format_confidence | 0.1000 | +0.0333 | 3 | 2 | 0 | 25 |
| rerank_target_on_low_format | 0.1333 | +0.0667 | 3 | 1 | 1 | 25 |
| rerank_agreement_or_target | 0.0333 | -0.0333 | 0 | 1 | 1 | 28 |

Interpretation:

Format-first reranking is the first non-oracle selector to test whether stochastic route candidates can be used without label leakage. Compare it to target-alone and to the oracle aggregate to separate candidate quality from selection quality.
