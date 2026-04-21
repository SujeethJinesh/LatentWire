# Toy Hub Sticky Frontier Stack

Deterministic composition test for shared hubs, routing stability, mixed-bit frontiers, and verifier stop rules.

| Method | Accuracy | MSE | Route acc | Route entropy | Route load | Perturb stability | Atom recovery | Avg stop steps | Over-refine | Bytes proxy | Compute proxy | Help vs raw pairwise | Harm vs raw pairwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_pairwise_bridge | 0.4722 | 2.6867 | 0.6944 | 2.2638 | 0.8403 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 27360.0000 | 415.8000 | 0.0000 | 0.0000 |
| monolithic_bridge | 0.4722 | 2.6867 | 0.6944 | 2.2638 | 0.8403 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1368.0000 | 351.0000 | 0.0000 | 0.0000 |
| hub_dictionary_only | 0.3056 | 3.1765 | 0.6944 | 2.2638 | 0.8403 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 16013.0000 | 376.9200 | 0.0000 | 0.1667 |
| hub_feature_router | 0.3056 | 3.1765 | 0.5278 | 2.2558 | 0.8403 | 0.8333 | 0.0000 | 1.0000 | 0.0000 | 16013.0000 | 376.9200 | 0.0000 | 0.1667 |
| hub_sticky_router | 0.3056 | 3.1765 | 0.4444 | 2.3140 | 0.9444 | 0.8333 | 0.0000 | 1.0000 | 0.0000 | 16013.0000 | 376.9200 | 0.0000 | 0.1667 |
| hub_sticky_protected_mixed_bit_frontier | 0.3056 | 3.1494 | 0.4444 | 2.3140 | 0.9444 | 0.8333 | 0.5417 | 1.0000 | 0.0000 | 16058.0000 | 462.7800 | 0.0000 | 0.1667 |
| hub_sticky_frontier_verifier_stop | 0.9444 | 0.0134 | 0.4444 | 2.3140 | 0.9444 | 0.8333 | 0.5417 | 1.8056 | 0.0000 | 16058.0000 | 510.4500 | 0.4722 | 0.0000 |
| random_router_control | 0.3056 | 3.1765 | 0.2222 | 2.2473 | 0.8403 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 16013.0000 | 376.9200 | 0.0000 | 0.1667 |
| confidence_router_control | 0.3056 | 3.1765 | 0.6944 | 2.2638 | 0.8403 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 16013.0000 | 376.9200 | 0.0000 | 0.1667 |
| oracle_router_control | 0.3056 | 3.1765 | 1.0000 | 2.2298 | 0.8056 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 16013.0000 | 376.9200 | 0.0000 | 0.1667 |

## Bit Histograms

- `raw_pairwise_bridge`: {"16": 432}
- `monolithic_bridge`: {"16": 432}
- `hub_dictionary_only`: {"3": 432}
- `hub_feature_router`: {"3": 432}
- `hub_sticky_router`: {"3": 432}
- `hub_sticky_protected_mixed_bit_frontier`: {"3": 180, "8": 144}
- `hub_sticky_frontier_verifier_stop`: {"3": 180, "8": 144}
- `random_router_control`: {"3": 432}
- `confidence_router_control`: {"3": 432}
- `oracle_router_control`: {"3": 432}

## Route Histograms

- `raw_pairwise_bridge`: {"0": 5, "1": 5, "2": 10, "3": 9, "4": 7}
- `monolithic_bridge`: {"0": 5, "1": 5, "2": 10, "3": 9, "4": 7}
- `hub_dictionary_only`: {"0": 5, "1": 5, "2": 10, "3": 9, "4": 7}
- `hub_feature_router`: {"0": 6, "1": 7, "2": 10, "3": 4, "4": 9}
- `hub_sticky_router`: {"0": 7, "1": 7, "2": 8, "3": 6, "4": 8}
- `hub_sticky_protected_mixed_bit_frontier`: {"0": 7, "1": 7, "2": 8, "3": 6, "4": 8}
- `hub_sticky_frontier_verifier_stop`: {"0": 7, "1": 7, "2": 8, "3": 6, "4": 8}
- `random_router_control`: {"0": 7, "1": 8, "2": 11, "3": 6, "4": 4}
- `confidence_router_control`: {"0": 5, "1": 5, "2": 10, "3": 9, "4": 7}
- `oracle_router_control`: {"0": 4, "1": 7, "2": 9, "3": 11, "4": 5}

## Stop Reasons

- `raw_pairwise_bridge`: {"direct_bridge": 36}
- `monolithic_bridge`: {"direct_bridge": 36}
- `hub_dictionary_only`: {"low_bit_only": 36}
- `hub_feature_router`: {"low_bit_only": 36}
- `hub_sticky_router`: {"low_bit_only": 36}
- `hub_sticky_protected_mixed_bit_frontier`: {"frontier_fixed": 36}
- `hub_sticky_frontier_verifier_stop`: {"confidence_reached": 12, "max_steps": 3, "verifier_harm": 21}
- `random_router_control`: {"low_bit_only": 36}
- `confidence_router_control`: {"low_bit_only": 36}
- `oracle_router_control`: {"low_bit_only": 36}

## Sources Consulted

- https://arxiv.org/abs/2502.03714
- https://arxiv.org/abs/2410.06981
- https://arxiv.org/abs/2506.14038
- https://arxiv.org/abs/2204.08396
- https://arxiv.org/abs/2001.00281
- https://arxiv.org/abs/2310.05175
- https://arxiv.org/abs/2311.14125
- https://arxiv.org/abs/2501.13122

## Interpretation

This toy composes a shared hub dictionary, route selection, sticky routing, protected mixed-bit frontiering, and verifier stopping on the same synthetic family-transfer problem. The key question is whether the stack adds gains or whether later stages interfere by erasing route stability, frontier coverage, or stop fidelity.
