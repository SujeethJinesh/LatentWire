# Candidate-Embedding Receiver Multiseed and Heldout Summary

- created UTC: `2026-04-29T19:43:37.161397+00:00`
- readout: 8-byte learned receiver passes 3/3 same-distribution seeds; 4-byte receiver fails 1/3 seeds; core-to-holdout transfer fails and is not promoted.

| Artifact | Train | Eval | N | Cand dims | Budget | Pass | Matched | Target | Best destructive | Delta control | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gated_budget4_seed29_30` | `all` | `all` | 512 | 32 | 4 | `True` | 0.748 | 0.250 | 0.262 | 0.486 | 0.998 |
| `gated_budget4_seed31_32` | `all` | `all` | 512 | 32 | 4 | `True` | 0.691 | 0.250 | 0.281 | 0.410 | 0.998 |
| `gated_budget4_seed37_38` | `all` | `all` | 512 | 32 | 4 | `False` | 0.328 | 0.250 | 0.279 | 0.049 | 1.000 |
| `diagnostic_budget8_seed29_30` | `all` | `all` | 512 | 32 | 8 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 1.000 |
| `diagnostic_budget8_seed31_32` | `all` | `all` | 512 | 32 | 8 | `True` | 0.514 | 0.250 | 0.283 | 0.230 | 1.000 |
| `diagnostic_budget8_seed37_38` | `all` | `all` | 512 | 32 | 8 | `True` | 0.859 | 0.250 | 0.275 | 0.584 | 1.000 |
| `heldout_core_to_holdout_budget8_seed29_30` | `core` | `holdout` | 512 | 32 | 8 | `False` | 0.453 | 0.250 | 0.311 | 0.143 | 0.809 |
| `heldout_core_to_holdout_budget8_seed29_30_no_candidate_feats_n256` | `core` | `holdout` | 256 | 0 | 8 | `False` | 0.332 | 0.250 | 0.309 | 0.023 | 0.742 |

## Same-Distribution Aggregates

| Budget | Pass count | Matched mean | Matched min | Max destructive | Min delta control |
|---:|---:|---:|---:|---:|---:|
| 4 | 2/3 | 0.589 | 0.328 | 0.281 | 0.049 |
| 8 | 3/3 | 0.749 | 0.514 | 0.283 | 0.230 |

## Interpretation

The learned receiver is now a real same-distribution technical contribution at 8 bytes, but it is not cross-family stable. The no-candidate-feature invariant ablation worsens heldout transfer, so the next method step should be an anchor-relative or fold-heldout receiver rather than simply deleting raw candidate features.
