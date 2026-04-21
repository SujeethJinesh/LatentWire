# Toy GPA Sparse Dictionary Hub Sweep

- seed: `13`
- dim: `18`
- atoms: `10`
- classes: `5`
- families: `5`
- held-out family: `4`
- seen shots / class: `20`
- held-out shot grid: `[1, 2, 4]`

This toy tests the next positive-method branch directly: GPA-style canonicalization first, then a sparse shared dictionary, then one verifier-gated repair step with routing frozen by construction.

References:
- Multi-Way Representation Alignment: https://arxiv.org/abs/2602.06205
- Universal Sparse Autoencoders: https://arxiv.org/abs/2502.03714
- Delta-Crosscoder: https://arxiv.org/abs/2603.04426

| Shot | Method | Accuracy | MSE | dAcc vs few-shot | dMSE vs few-shot | Atom rec | Dead atoms | Perplexity | Repair accept | Repair help | Repair harm | Shared Basis |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0000 | heldout_fewshot_ridge | 1.0000 | 0.1825 | 0.0000 | 0.0000 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 1.0000 | global_seen_ridge | 0.1300 | 1.0624 | -0.8700 | 0.8799 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 1.0000 | multiway_gpa_canonical | 1.0000 | 0.2355 | 0.0000 | 0.0530 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 1.0000 | multiway_gpa_sparse_dictionary | 1.0000 | 0.1171 | 0.0000 | -0.0654 | 0.1900 | 0.0000 | 9.9996 | 0.0000 | 0.0000 | 0.0000 | True |
| 1.0000 | multiway_gpa_sparse_dictionary_repair | 1.0000 | 0.1171 | 0.0000 | -0.0654 | 0.1900 | 0.0000 | 9.9996 | 0.0000 | 0.0000 | 0.0000 | True |
| 1.0000 | oracle_family_ridge | 1.0000 | 0.0019 | 0.0000 | -0.1806 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 2.0000 | heldout_fewshot_ridge | 1.0000 | 0.0736 | 0.0000 | 0.0000 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 2.0000 | global_seen_ridge | 0.1300 | 1.0624 | -0.8700 | 0.9888 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 2.0000 | multiway_gpa_canonical | 1.0000 | 0.1153 | 0.0000 | 0.0417 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 2.0000 | multiway_gpa_sparse_dictionary | 1.0000 | 0.1195 | 0.0000 | 0.0459 | 0.1900 | 0.0000 | 9.9996 | 0.0000 | 0.0000 | 0.0000 | True |
| 2.0000 | multiway_gpa_sparse_dictionary_repair | 1.0000 | 0.1195 | 0.0000 | 0.0459 | 0.1900 | 0.0000 | 9.9996 | 0.0000 | 0.0000 | 0.0000 | True |
| 2.0000 | oracle_family_ridge | 1.0000 | 0.0019 | 0.0000 | -0.0717 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 4.0000 | heldout_fewshot_ridge | 1.0000 | 0.0158 | 0.0000 | 0.0000 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 4.0000 | global_seen_ridge | 0.1300 | 1.0624 | -0.8700 | 1.0466 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |
| 4.0000 | multiway_gpa_canonical | 1.0000 | 0.1697 | 0.0000 | 0.1539 | - | - | - | 0.0000 | 0.0000 | 0.0000 | True |
| 4.0000 | multiway_gpa_sparse_dictionary | 1.0000 | 0.1186 | 0.0000 | 0.1029 | 0.1900 | 0.0000 | 9.9996 | 0.0000 | 0.0000 | 0.0000 | True |
| 4.0000 | multiway_gpa_sparse_dictionary_repair | 1.0000 | 0.1186 | 0.0000 | 0.1029 | 0.1900 | 0.0000 | 9.9996 | 0.0000 | 0.0000 | 0.0000 | True |
| 4.0000 | oracle_family_ridge | 1.0000 | 0.0019 | 0.0000 | -0.0138 | - | - | - | 0.0000 | 0.0000 | 0.0000 | False |

## Best Non-Oracle by Shot
- 1 shot/class: best accuracy `multiway_gpa_sparse_dictionary` (1.0000 acc, 0.1171 MSE); lowest non-oracle MSE `multiway_gpa_sparse_dictionary` (0.1171); best shared-basis `multiway_gpa_sparse_dictionary` (0.1171 MSE)
- 2 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0736 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0736); best shared-basis `multiway_gpa_canonical` (0.1153 MSE)
- 4 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0158 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0158); best shared-basis `multiway_gpa_sparse_dictionary` (0.1186 MSE)
