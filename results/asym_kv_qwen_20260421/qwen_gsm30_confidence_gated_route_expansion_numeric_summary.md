# Confidence-Gated Route Expansion Summary

- Selection policy: `numeric_consistency_then_completion`
- Calibration examples: `15`
- Eval examples: `15`
- Low threshold: `5.8750`
- High threshold: `5.8750`
- Train accuracy: `0.0667`
- Train avg seed budget: `1.4000`

| Method | Accuracy | Delta vs target | Avg seed budget | Oracle gap | Target selected |
|---|---:|---:|---:|---:|---:|
| target_alone | 0.0667 | +0.0000 | - | - | - |
| fixed_route_budget_0 | 0.0667 | +0.0000 | 0.0000 | 0.2000 | 1.0000 |
| fixed_route_budget_1 | 0.1333 | +0.0667 | 1.0000 | 0.1333 | 0.6000 |
| fixed_route_budget_3 | 0.1333 | +0.0667 | 3.0000 | 0.1333 | 0.2667 |
| random_route_budget_matched | 0.0000 | -0.0667 | 1.0000 | 0.2667 | 0.8000 |
| confidence_gated_route_expansion | 0.1333 | +0.0667 | 2.2000 | 0.1333 | 0.5333 |

## Confidence-Gated Subgroups

| Group | Count | Accuracy | Avg seed budget | Full oracle accuracy | Target selected | Proxy range |
|---|---:|---:|---:|---:|---:|---:|
| target_proxy_0 | 1 | 0.0000 | 3.0000 | 0.0000 | 0.0000 | -1.25--1.25 |
| target_proxy_2 | 14 | 0.1429 | 2.1429 | 0.2857 | 0.5714 | 5.50-9.50 |
