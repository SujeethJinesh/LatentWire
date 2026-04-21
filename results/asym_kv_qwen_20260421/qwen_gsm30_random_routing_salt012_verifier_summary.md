# Target-Model Verifier Reranker Summary

| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| rerank_target_model_verifier | 0.0667 | +0.0000 | 0 | 0 | 2 | 28 | 0.0000 |

Interpretation:

This is a non-oracle target-model listwise selector over the same stochastic candidate set. The raw verifier response and fallback flag are logged per example so selection failures can be audited.
