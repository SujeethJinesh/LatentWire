# Confidence-Gated Route Expansion Summary

- Selection policy: `target_on_strict_format`
- Calibration examples: `15`
- Eval examples: `15`
- Low threshold: `5.6250`
- High threshold: `6.5000`
- Train accuracy: `0.1333`
- Train avg seed budget: `1.5333`

| Method | Accuracy | Delta vs target | Avg seed budget | Oracle gap | Target selected |
|---|---:|---:|---:|---:|---:|
| target_alone | 0.0667 | +0.0000 | - | - | - |
| fixed_route_budget_0 | 0.0667 | +0.0000 | 0.0000 | 0.2000 | 1.0000 |
| fixed_route_budget_1 | 0.2000 | +0.1333 | 1.0000 | 0.0667 | 0.8667 |
| fixed_route_budget_3 | 0.2000 | +0.1333 | 3.0000 | 0.0667 | 0.6667 |
| random_route_budget_matched | 0.1333 | +0.0667 | 1.3333 | 0.1333 | 0.9333 |
| confidence_gated_route_expansion | 0.2000 | +0.1333 | 2.3333 | 0.0667 | 0.6667 |

## Confidence-Gated Subgroups

| Group | Count | Accuracy | Avg seed budget | Full oracle accuracy | Target selected | Proxy range |
|---|---:|---:|---:|---:|---:|---:|
| target_proxy_0 | 1 | 0.0000 | 3.0000 | 0.0000 | 0.0000 | -1.25--1.25 |
| target_proxy_2 | 14 | 0.2143 | 2.2857 | 0.2857 | 0.7143 | 5.50-9.50 |
