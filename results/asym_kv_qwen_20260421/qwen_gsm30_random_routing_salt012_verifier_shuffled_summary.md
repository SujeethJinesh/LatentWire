# Target-Model Verifier Reranker Summary

| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong | Fallback rate | Target selected | Choice A | Target was A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rerank_target_model_verifier | 0.1000 | +0.0333 | 2 | 1 | 1 | 26 | 0.0000 | 0.2000 | 0.9667 | 0.2333 |

Interpretation:

This is a non-oracle target-model listwise selector over the same stochastic candidate set. The raw verifier response, label order, fallback flag, and position-bias rates are logged per example so selection failures can be audited.
