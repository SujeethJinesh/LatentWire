# Stochastic Route Aggregation Summary

| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| target_alone | 0.0667 | +0.0000 | 0 | 0 | 0 | 0 |
| stochastic_majority_vote | 0.0333 | -0.0333 | 1 | 2 | 0 | 27 |
| stochastic_target_tiebreak | 0.0333 | -0.0333 | 0 | 1 | 1 | 28 |
| stochastic_any_seed_oracle | 0.2667 | +0.2000 | 7 | 1 | 1 | 21 |
| stochastic_target_or_seed_oracle | 0.3000 | +0.2333 | 7 | 0 | 2 | 21 |

Interpretation:

The oracle rows measure candidate-set quality, while majority and target tie-break measure naive non-oracle selection. A large oracle gap with weak naive selection means the next blocker is verifier/reranker selection rather than raw sampling.
