# Toy Confidence-Gated Compute

- Seed: `0`
- Train examples: `384`
- Test examples: `192`
- Calibrated thresholds: low `0.3047`, high `0.5379`
- Train accuracy: `0.6693`
- Train avg budget: `2.0208`

| Method | Accuracy | Avg budget | Compute fraction | Oracle gap | Probe ECE | Selected ECE | Probe AUROC | Selected AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_budget_1 | 0.6146 | 1.0000 | 0.2500 | 0.3646 | 0.1392 | 0.1392 | 0.7213 | 0.7213 |
| fixed_budget_2 | 0.6406 | 2.0000 | 0.5000 | 0.3385 | 0.1392 | 0.0970 | 0.7213 | 0.6757 |
| fixed_budget_4 | 0.6458 | 4.0000 | 1.0000 | 0.3333 | 0.1392 | 0.0964 | 0.7213 | 0.6396 |
| random_budget_matched | 0.6250 | 1.9740 | 0.4935 | 0.3542 | 0.1392 | 0.1041 | 0.7213 | 0.6873 |
| confidence_gated | 0.6510 | 2.0312 | 0.5078 | 0.3281 | 0.1392 | 0.1027 | 0.7213 | 0.6746 |

## Confidence-Gated Subgroups

| Group | Count | Accuracy | Avg budget | Oracle gap | Probe ECE | Selected ECE |
|---|---:|---:|---:|---:|---:|---:|
| difficulty_0 | 64 | 0.7344 | 2.0312 | 0.2188 | 0.2646 | 0.1646 |
| difficulty_1 | 64 | 0.6406 | 2.1875 | 0.3438 | 0.1531 | 0.1065 |
| difficulty_2 | 64 | 0.5781 | 1.8750 | 0.4219 | 0.1262 | 0.1191 |

| Probe group | Count | Accuracy | Avg budget | Oracle gap | Probe ECE | Selected ECE |
|---|---:|---:|---:|---:|---:|---:|
| probe_confidence_0 | 64 | 0.4844 | 2.9375 | 0.4531 | 0.0896 | 0.1353 |
| probe_confidence_1 | 64 | 0.6406 | 2.0000 | 0.3594 | 0.2061 | 0.1255 |
| probe_confidence_2 | 64 | 0.8281 | 1.1562 | 0.1719 | 0.1603 | 0.1655 |
