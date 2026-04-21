# Toy Routed Projector Bank

Deterministic multimodal-inspired ablation for route-specific projector banks in cross-model latent exchange.

| Method | Accuracy | MSE | Route acc | Route entropy | Route stability | Expert utilization | Bytes proxy | Compute proxy | Help | Harm | MSE help | MSE harm |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| no_route_baseline | 0.2688 | 1.9278 | 0.2188 | 0.0000 | 1.0000 | `0:1.00, 1:0.00, 2:0.00, 3:0.00` | 80.0000 | 20.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| monolithic_projector | 0.8687 | 0.1002 | 0.2188 | 0.0000 | 1.0000 | `0:1.00, 1:0.00, 2:0.00, 3:0.00` | 1680.0000 | 400.0000 | 0.6812 | 0.0812 | 1.0000 | 0.0000 |
| oracle_routed_bank | 0.9688 | 0.0031 | 1.0000 | 1.9846 | 1.0000 | `0:0.22, 1:0.31, 2:0.24, 3:0.22` | 6720.0000 | 404.0000 | 0.7188 | 0.0188 | 1.0000 | 0.0000 |
| confidence_routed_bank | 0.3000 | 2.0974 | 0.1437 | 1.9895 | 0.9563 | `0:0.22, 1:0.29, 2:0.22, 3:0.26` | 6760.0000 | 1600.0000 | 0.2313 | 0.2000 | 0.3875 | 0.6125 |
| feature_routed_bank | 0.9187 | 0.1387 | 0.9187 | 1.9960 | 1.0000 | `0:0.24, 1:0.23, 2:0.24, 3:0.28` | 7040.0000 | 480.0000 | 0.7000 | 0.0500 | 0.9187 | 0.0812 |
| random_routed_bank | 0.4375 | 1.7458 | 0.2500 | 1.9959 | 1.0000 | `0:0.23, 1:0.27, 2:0.27, 3:0.23` | 6720.0000 | 400.0000 | 0.3375 | 0.1688 | 0.4812 | 0.5188 |

## Failure Tags

- `no_route_baseline`: lost_baseline_correct=0, ok_correct=43, other=0, projection_error=27, route_mismatch=82, wrong_low_mse=8
- `monolithic_projector`: lost_baseline_correct=0, ok_correct=139, other=0, projection_error=0, route_mismatch=21, wrong_low_mse=0
- `oracle_routed_bank`: lost_baseline_correct=3, ok_correct=155, other=0, projection_error=1, route_mismatch=0, wrong_low_mse=1
- `confidence_routed_bank`: lost_baseline_correct=0, ok_correct=48, other=0, projection_error=0, route_mismatch=112, wrong_low_mse=0
- `feature_routed_bank`: lost_baseline_correct=3, ok_correct=147, other=0, projection_error=0, route_mismatch=9, wrong_low_mse=1
- `random_routed_bank`: lost_baseline_correct=0, ok_correct=70, other=0, projection_error=0, route_mismatch=90, wrong_low_mse=0

## Interpretation

A routed projector bank is useful when source latents occupy route-specific gauges. Oracle routing upper-bounds the bank; feature routing tests cheap centroid routing; confidence routing tests target-head self-selection; random routing isolates bank capacity from route quality.
