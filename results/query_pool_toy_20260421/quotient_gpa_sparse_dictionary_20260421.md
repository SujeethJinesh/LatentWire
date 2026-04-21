# Toy Quotient + GPA Sparse Dictionary Sweep

- seed: `7`
- heads: `4`
- head dim: `6`
- atoms: `10`
- classes: `5`
- families: `5`
- held-out family: `4`
- anchor family: `0`
- seen shots / class: `20`
- held-out shot grid: `[1, 2, 4]`

This toy composes three low-shot ideas directly: quotient-aware head matching, GPA-style multi-family canonicalization, and a sparse shared dictionary. The goal is to see whether the low-shot symmetry gain and the low-shot shared-basis gain are additive.

References:
- Complete Characterization of Gauge Symmetries in Transformer Architectures: https://openreview.net/forum?id=KrkbYbK0cH
- Multi-Way Representation Alignment: https://arxiv.org/abs/2602.06205
- Universal Sparse Autoencoders: https://arxiv.org/abs/2502.03714

| Shot | Method | Accuracy | MSE | dAcc vs few-shot | dMSE vs few-shot | Centroid cosine | Gauge residual | Head-match acc | Canonical gap | Atom recovery | Dead atom rate | Codebook perplexity | Repair accept | Repair help | Repair harm | Shared Basis |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0000 | heldout_fewshot_ridge | 1.0000 | 0.1003 | 0.0000 | 0.0000 | 0.9473 | 0.0416 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 1.0000 | global_seen_ridge | 0.4150 | 0.5850 | -0.5850 | 0.4847 | 0.5270 | 0.4099 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 1.0000 | quotient_match_after_fix | 1.0000 | 0.0693 | 0.0000 | -0.0310 | 0.9643 | 0.1750 | 1.0000 | 0.1750 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 1.0000 | quotient_gpa_canonical | 1.0000 | 0.0637 | 0.0000 | -0.0366 | 0.9962 | 0.1750 | 1.0000 | 0.0051 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 1.0000 | quotient_gpa_sparse_dictionary | 1.0000 | 0.0568 | 0.0000 | -0.0435 | 0.9975 | 0.1750 | 1.0000 | 0.4798 | 0.1950 | 0.0000 | 9.9997 | 0.0000 | 0.0000 | 0.0000 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_repair | 1.0000 | 0.0568 | 0.0000 | -0.0435 | 0.9975 | 0.1750 | 1.0000 | 0.4798 | 0.1950 | 0.0000 | 9.9997 | 0.0000 | 0.0000 | 0.0000 | True |
| 1.0000 | oracle_family_ridge | 1.0000 | 0.0012 | 0.0000 | -0.0991 | 0.9999 | 0.0152 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 2.0000 | heldout_fewshot_ridge | 1.0000 | 0.0638 | 0.0000 | 0.0000 | 0.9733 | 0.0321 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 2.0000 | global_seen_ridge | 0.4150 | 0.5850 | -0.5850 | 0.5212 | 0.5270 | 0.4099 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 2.0000 | quotient_match_after_fix | 1.0000 | 0.0597 | 0.0000 | -0.0041 | 0.9785 | 0.1070 | 1.0000 | 0.1070 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 2.0000 | quotient_gpa_canonical | 1.0000 | 0.0738 | 0.0000 | 0.0099 | 0.9962 | 0.1070 | 1.0000 | 0.0051 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 2.0000 | quotient_gpa_sparse_dictionary | 1.0000 | 0.0576 | 0.0000 | -0.0062 | 0.9975 | 0.1070 | 1.0000 | 0.4798 | 0.1950 | 0.0000 | 9.9997 | 0.0000 | 0.0000 | 0.0000 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_repair | 1.0000 | 0.0576 | 0.0000 | -0.0062 | 0.9975 | 0.1070 | 1.0000 | 0.4798 | 0.1950 | 0.0000 | 9.9997 | 0.0000 | 0.0000 | 0.0000 | True |
| 2.0000 | oracle_family_ridge | 1.0000 | 0.0012 | 0.0000 | -0.0626 | 0.9999 | 0.0152 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 4.0000 | heldout_fewshot_ridge | 1.0000 | 0.0179 | 0.0000 | 0.0000 | 0.9965 | 0.0185 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 4.0000 | global_seen_ridge | 0.4150 | 0.5850 | -0.5850 | 0.5671 | 0.5270 | 0.4099 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 4.0000 | quotient_match_after_fix | 1.0000 | 0.0531 | 0.0000 | 0.0352 | 0.9861 | 0.0584 | 1.0000 | 0.0584 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 4.0000 | quotient_gpa_canonical | 1.0000 | 0.0607 | 0.0000 | 0.0428 | 0.9962 | 0.0584 | 1.0000 | 0.0051 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 4.0000 | quotient_gpa_sparse_dictionary | 1.0000 | 0.0555 | 0.0000 | 0.0376 | 0.9975 | 0.0584 | 1.0000 | 0.4798 | 0.1950 | 0.0000 | 9.9997 | 0.0000 | 0.0000 | 0.0000 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_repair | 1.0000 | 0.0555 | 0.0000 | 0.0376 | 0.9975 | 0.0584 | 1.0000 | 0.4798 | 0.1950 | 0.0000 | 9.9997 | 0.0000 | 0.0000 | 0.0000 | True |
| 4.0000 | oracle_family_ridge | 1.0000 | 0.0012 | 0.0000 | -0.0167 | 0.9999 | 0.0152 | - | - | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |

## Best Non-Oracle by Shot
- 1 shot/class: best accuracy `quotient_gpa_sparse_dictionary` (1.0000 acc, 0.0568 MSE); lowest non-oracle MSE `quotient_gpa_sparse_dictionary` (0.0568); best shared-basis `quotient_gpa_sparse_dictionary` (0.0568 MSE)
- 2 shot/class: best accuracy `quotient_gpa_sparse_dictionary` (1.0000 acc, 0.0576 MSE); lowest non-oracle MSE `quotient_gpa_sparse_dictionary` (0.0576); best shared-basis `quotient_gpa_sparse_dictionary` (0.0576 MSE)
- 4 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0179 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0179); best shared-basis `quotient_match_after_fix` (0.0531 MSE)
