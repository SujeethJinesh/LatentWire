# Family-Invariant Receiver Follow-Up

- created UTC: `2026-04-29T20:19:01.279972+00:00`
- readout: Simple coordinate-free and anchor-relative receiver variants do not fix core-to-holdout transfer; the naive anchor bank is pruned.

| Artifact | Receiver | Packet features | Packet dim | Cand dims | N | Pass | Matched | Target | Best destructive | Delta control | Oracle |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `heldout_core_to_holdout_budget8_seed29_30` | `ridge` | `hashed` | 512 | 32 | 512 | `False` | 0.453 | 0.250 | 0.311 | 0.143 | 0.809 |
| `heldout_core_to_holdout_code_similarity_budget8_seed29_30` | `code_similarity` | `hashed` | 512 | 0 | 512 | `False` | 0.256 | 0.250 | 0.285 | -0.029 | 1.000 |
| `heldout_core_to_holdout_anchor_relative_code_similarity_budget8_seed29_30` | `code_similarity` | `anchor_relative` | 128 | 0 | 512 | `False` | 0.281 | 0.250 | 0.258 | 0.023 | 0.756 |
| `heldout_core_to_holdout_anchor_relative_ridge_budget8_seed29_30` | `ridge` | `anchor_relative` | 128 | 0 | 512 | `False` | 0.303 | 0.250 | 0.438 | -0.135 | 0.342 |

## Interpretation

The hashed code-similarity row has perfect oracle decoding but target-level matched accuracy, so the source encoder is not producing transferable candidate-code packets across families. The anchor-relative code-similarity row keeps controls clean but reduces oracle headroom, and the anchor-relative ridge row lets controls dominate. This prunes the simple cosine-anchor fix and points to fold-heldout calibration or sparse shared dictionaries.
