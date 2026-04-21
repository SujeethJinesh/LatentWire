# Toy Route-Class Frontier Sweep

Route-conditioned patch-effect frontier and stop sweep for the hub stack.

| Method | Router | Frontier | Stop | Accuracy | Delta vs raw | Route acc | Patch corr | Protect-oracle overlap | Selected atoms | Protected atoms | Avg stop steps | Over-refine | Bytes proxy | Compute proxy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conditional_prior_base | conditional_prior | all_low_bit | none | 0.6250 | -0.1094 | 0.6979 | 0.1947 | 0.0000 | 12.0000 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 530.0000 |
| conditional_prior_quant_error_frontier | conditional_prior | quant_error_frontier | none | 0.6354 | -0.0990 | 0.6979 | 0.1947 | 0.5130 | 9.0000 | 4.0000 | 1.0000 | 0.0000 | 22964.0000 | 515.0000 |
| conditional_prior_route_class_patch_protect | conditional_prior | route_class_patch_protect | none | 0.6354 | -0.0990 | 0.6979 | 0.1947 | 0.4688 | 12.0000 | 4.0000 | 1.0000 | 0.0000 | 22964.0000 | 530.0000 |
| conditional_prior_route_class_patch_frontier | conditional_prior | route_class_patch_frontier | none | 0.6146 | -0.1198 | 0.6979 | 0.1947 | 0.4688 | 9.0000 | 4.0000 | 1.0000 | 0.0000 | 22964.0000 | 515.0000 |
| conditional_prior_route_class_patch_frontier_stop | conditional_prior | route_class_patch_frontier | route_class_mode_step | 0.6146 | -0.1198 | 0.6979 | 0.1947 | 0.4688 | 9.0000 | 4.0000 | 1.9688 | 0.1927 | 22964.0000 | 544.0625 |
| oracle_base | oracle_router | all_low_bit | none | 0.8229 | 0.0885 | 1.0000 | 0.2836 | 0.0000 | 12.0000 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 570.0000 |
| oracle_quant_error_frontier | oracle_router | quant_error_frontier | none | 0.8125 | 0.0781 | 1.0000 | 0.2836 | 0.5208 | 9.0000 | 4.0000 | 1.0000 | 0.0000 | 22964.0000 | 555.0000 |
| oracle_route_class_patch_protect | oracle_router | route_class_patch_protect | none | 0.8125 | 0.0781 | 1.0000 | 0.2836 | 0.4701 | 12.0000 | 4.0000 | 1.0000 | 0.0000 | 22964.0000 | 570.0000 |
| oracle_route_class_patch_frontier | oracle_router | route_class_patch_frontier | none | 0.7969 | 0.0625 | 1.0000 | 0.2836 | 0.4701 | 9.0000 | 4.0000 | 1.0000 | 0.0000 | 22964.0000 | 555.0000 |
| oracle_route_class_patch_frontier_stop | oracle_router | route_class_patch_frontier | route_class_mode_step | 0.7969 | 0.0625 | 1.0000 | 0.2836 | 0.4701 | 9.0000 | 4.0000 | 1.9583 | 0.2292 | 22964.0000 | 583.7500 |

## Interpretation

This sweep tests whether a route-/class-calibrated patch-effect frontier repairs the negative frontier result from the previous hub sweep. If route_class_patch variants improve patch-rank correlation and accuracy relative to the quant-error frontier, the next paper lane should replace the current frontier heuristic rather than abandoning protected communication altogether.

## Sources Consulted

- results/query_pool_toy_20260421/protected_frontier_selection_20260421.md
- results/query_pool_toy_20260421/hub_router_frontier_sweep_20260421.md
- https://arxiv.org/abs/2502.03714
- https://arxiv.org/abs/2506.14038
