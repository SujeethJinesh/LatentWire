# Toy Multi-Way Canonical Hub Sweep

- seed: `11`
- dim: `18`
- classes: `5`
- families: `5`
- held-out family: `4`
- seen shots / class: `20`
- held-out shot grid: `[1, 2, 4, 8]`

This toy isolates whether a multi-way canonical hub helps when one family has only a few paired examples. It is the direct follow-up to the hub-router sweeps: fix the shared basis first, then revisit communication controls.

References:
- Multi-Way Representation Alignment: https://arxiv.org/abs/2602.06205
- Model Stitching: https://arxiv.org/abs/2303.11277

| Shot | Method | Accuracy | MSE | dAcc vs few-shot | dMSE vs few-shot | Centroid Cos | Canonical Gap | Shared Basis |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | heldout_fewshot_ridge | 1.0000 | 0.1463 | +0.0000 | +0.0000 | 0.9176 | 0.0779 | False |
| 1 | global_seen_ridge | 0.0781 | 0.8006 | -0.9219 | +0.6543 | 0.1287 | 0.6304 | True |
| 1 | anchor_family_transfer | 1.0000 | 0.2822 | +0.0000 | +0.1359 | 0.8877 | 0.2562 | True |
| 1 | multiway_gpa_canonical | 1.0000 | 0.1327 | +0.0000 | -0.0136 | 0.9536 | 0.2580 | True |
| 1 | oracle_family_ridge | 1.0000 | 0.0015 | +0.0000 | -0.1447 | 0.9999 | 0.0039 | True |
| 2 | heldout_fewshot_ridge | 1.0000 | 0.0753 | +0.0000 | +0.0000 | 0.9690 | 0.0301 | False |
| 2 | global_seen_ridge | 0.0781 | 0.8006 | -0.9219 | +0.7253 | 0.1287 | 0.6304 | True |
| 2 | anchor_family_transfer | 1.0000 | 0.2177 | +0.0000 | +0.1425 | 0.9443 | 0.1239 | True |
| 2 | multiway_gpa_canonical | 1.0000 | 0.1050 | +0.0000 | +0.0297 | 0.9793 | 0.1221 | True |
| 2 | oracle_family_ridge | 1.0000 | 0.0015 | +0.0000 | -0.0737 | 0.9999 | 0.0039 | True |
| 4 | heldout_fewshot_ridge | 1.0000 | 0.0144 | +0.0000 | +0.0000 | 0.9973 | 0.0061 | False |
| 4 | global_seen_ridge | 0.0781 | 0.8006 | -0.9219 | +0.7862 | 0.1287 | 0.6304 | True |
| 4 | anchor_family_transfer | 1.0000 | 0.1988 | +0.0000 | +0.1844 | 0.9644 | 0.0763 | True |
| 4 | multiway_gpa_canonical | 1.0000 | 0.1111 | +0.0000 | +0.0968 | 0.9794 | 0.0766 | True |
| 4 | oracle_family_ridge | 1.0000 | 0.0015 | +0.0000 | -0.0129 | 0.9999 | 0.0039 | True |
| 8 | heldout_fewshot_ridge | 1.0000 | 0.0023 | +0.0000 | +0.0000 | 0.9997 | 0.0038 | False |
| 8 | global_seen_ridge | 0.0781 | 0.8006 | -0.9219 | +0.7983 | 0.1287 | 0.6304 | True |
| 8 | anchor_family_transfer | 1.0000 | 0.1982 | +0.0000 | +0.1960 | 0.9736 | 0.0477 | True |
| 8 | multiway_gpa_canonical | 1.0000 | 0.0985 | +0.0000 | +0.0962 | 0.9873 | 0.0485 | True |
| 8 | oracle_family_ridge | 1.0000 | 0.0015 | +0.0000 | -0.0008 | 0.9999 | 0.0039 | True |

## Best Non-Oracle by Shot
- 1 shot/class: best accuracy `multiway_gpa_canonical` (1.0000 acc, 0.1327 MSE); lowest non-oracle MSE `multiway_gpa_canonical` (0.1327); best shared-basis `multiway_gpa_canonical` (1.0000 acc, 0.1327 MSE)
- 2 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0753 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0753); best shared-basis `multiway_gpa_canonical` (1.0000 acc, 0.1050 MSE)
- 4 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0144 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0144); best shared-basis `multiway_gpa_canonical` (1.0000 acc, 0.1111 MSE)
- 8 shot/class: best accuracy `heldout_fewshot_ridge` (1.0000 acc, 0.0023 MSE); lowest non-oracle MSE `heldout_fewshot_ridge` (0.0023); best shared-basis `multiway_gpa_canonical` (1.0000 acc, 0.0985 MSE)
