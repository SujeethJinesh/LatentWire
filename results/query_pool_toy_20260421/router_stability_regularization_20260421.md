# Toy Router Stability Regularization

This deterministic ablation tests whether projector-bank routing is accurate, balanced, and stable under a paraphrase-like perturbation.

| Method | Accuracy | MSE | Route acc | Route entropy | Gate entropy | Load balance | Collapse | Perturb stable | Utilization | Bytes | Compute | Help | Harm | MSE help | MSE harm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard_feature_routing | 0.9438 | 0.0243 | 0.9875 | 1.9478 | 0.0000 | 0.8250 | 0.0917 | 0.9500 | 0:0.32, 1:0.31, 2:0.20, 3:0.17 | 16.0 | 80.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| confidence_routing | 0.3688 | 1.5497 | 0.2812 | 1.9112 | 0.0000 | 0.8000 | 0.2000 | 0.5437 | 0:0.40, 1:0.17, 2:0.24, 3:0.19 | 16.0 | 80.0 | 0.0063 | 0.5813 | 0.0000 | 0.7063 |
| smoothed_dense_routing | 0.7625 | 0.1008 | 0.9875 | 1.9478 | 1.0924 | 0.8250 | 0.0917 | 0.9500 | 0:0.32, 1:0.31, 2:0.20, 3:0.17 | 19.0 | 320.0 | 0.0125 | 0.1937 | 0.0125 | 0.9875 |
| load_balanced_routing | 0.9187 | 0.0905 | 0.9563 | 1.9693 | 0.0000 | 0.8667 | 0.0917 | 0.9125 | 0:0.32, 1:0.28, 2:0.20, 3:0.20 | 16.0 | 80.0 | 0.0000 | 0.0250 | 0.0000 | 0.0312 |
| sticky_paraphrase_stable_routing | 0.9438 | 0.0243 | 0.9875 | 1.9478 | 0.0000 | 0.8250 | 0.0917 | 1.0000 | 0:0.32, 1:0.31, 2:0.20, 3:0.17 | 16.0 | 160.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_routing | 0.3438 | 1.4314 | 0.2812 | 1.9959 | 0.0000 | 0.9500 | 0.0250 | 0.2688 | 0:0.23, 1:0.27, 2:0.27, 3:0.23 | 16.0 | 80.0 | 0.0000 | 0.6000 | 0.0063 | 0.7063 |
| oracle_routing | 0.9563 | 0.0031 | 1.0000 | 1.9496 | 0.0000 | 0.8250 | 0.0917 | 1.0000 | 0:0.32, 1:0.31, 2:0.19, 3:0.18 | 16.0 | 80.0 | 0.0125 | 0.0000 | 0.0125 | 0.0000 |

## Failure Tags

- `hard_feature_routing`: router_collapse=0, unstable_under_perturbation=8, ok_correct=144, route_mismatch=1, projection_error=3, lost_hard_correct=0, other_error=4
- `confidence_routing`: router_collapse=101, unstable_under_perturbation=39, ok_correct=20, route_mismatch=0, projection_error=0, lost_hard_correct=0, other_error=0
- `smoothed_dense_routing`: router_collapse=0, unstable_under_perturbation=8, ok_correct=120, route_mismatch=1, projection_error=30, lost_hard_correct=1, other_error=0
- `load_balanced_routing`: router_collapse=0, unstable_under_perturbation=14, ok_correct=139, route_mismatch=1, projection_error=3, lost_hard_correct=0, other_error=3
- `sticky_paraphrase_stable_routing`: router_collapse=0, unstable_under_perturbation=0, ok_correct=151, route_mismatch=2, projection_error=3, lost_hard_correct=0, other_error=4
- `random_routing`: router_collapse=0, unstable_under_perturbation=117, ok_correct=19, route_mismatch=24, projection_error=0, lost_hard_correct=0, other_error=0
- `oracle_routing`: router_collapse=0, unstable_under_perturbation=0, ok_correct=153, route_mismatch=0, projection_error=3, lost_hard_correct=0, other_error=4

## Reading

- `confidence_routing` is intentionally uncalibrated; it tests confidence collapse rather than projector quality.
- `smoothed_dense_routing` spends more compute/route bytes to avoid brittle one-hot routing.
- `load_balanced_routing` exposes the specialization-vs-utilization tradeoff.
- `sticky_paraphrase_stable_routing` tests whether route assignments survive small semantic-preserving perturbations.
