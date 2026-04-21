# Toy Hub Sticky Frontier Stack

Deterministic composition test for shared hubs, routing stability, mixed-bit frontiers, and verifier stop rules.

| Method | Accuracy | MSE | Route acc | Route entropy | Route load | Perturb stability | Atom recovery | Avg stop steps | Over-refine | Bytes proxy | Compute proxy | Help vs raw pairwise | Harm vs raw pairwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_pairwise_bridge | 0.7344 | 0.5842 | 0.6979 | 2.5695 | 0.9250 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 50400.0000 | 510.0000 | 0.0000 | 0.0000 |
| monolithic_bridge | 0.3698 | 1.1515 | 0.6979 | 2.5695 | 0.9250 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1680.0000 | 430.0000 | 0.0000 | 0.3646 |
| hub_dictionary_only | 0.6250 | 0.8243 | 0.6979 | 2.5695 | 0.9250 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 462.0000 | 0.0000 | 0.1094 |
| hub_feature_router | 0.6094 | 0.9477 | 0.6302 | 2.5718 | 0.9312 | 0.9010 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 462.0000 | 0.0000 | 0.1250 |
| hub_sticky_router | 0.6042 | 0.9626 | 0.6250 | 2.5670 | 0.9312 | 0.9219 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 462.0000 | 0.0000 | 0.1302 |
| hub_sticky_protected_mixed_bit_frontier | 0.5990 | 0.9283 | 0.6250 | 2.5670 | 0.9312 | 0.9219 | 0.5286 | 1.0000 | 0.0000 | 22964.0000 | 563.0000 | 0.0000 | 0.1354 |
| hub_sticky_frontier_verifier_stop | 0.5938 | 1.1173 | 0.6250 | 2.5670 | 0.9312 | 0.9219 | 0.5286 | 2.9688 | 0.4583 | 22964.0000 | 654.0625 | 0.0000 | 0.1406 |
| random_router_control | 0.3177 | 1.8793 | 0.1875 | 2.5744 | 0.9375 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 462.0000 | 0.0000 | 0.4167 |
| confidence_router_control | 0.6250 | 0.8243 | 0.6979 | 2.5695 | 0.9250 | 0.9896 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 462.0000 | 0.0000 | 0.1094 |
| oracle_router_control | 0.8229 | 0.2609 | 1.0000 | 2.5552 | 0.8938 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 22914.0000 | 462.0000 | 0.0885 | 0.0000 |

## Bit Histograms

- `raw_pairwise_bridge`: {"16": 2304}
- `monolithic_bridge`: {"16": 2304}
- `hub_dictionary_only`: {"3": 2304}
- `hub_feature_router`: {"3": 2304}
- `hub_sticky_router`: {"3": 2304}
- `hub_sticky_protected_mixed_bit_frontier`: {"3": 960, "8": 768}
- `hub_sticky_frontier_verifier_stop`: {"3": 960, "8": 768}
- `random_router_control`: {"3": 2304}
- `confidence_router_control`: {"3": 2304}
- `oracle_router_control`: {"3": 2304}

## Route Histograms

- `raw_pairwise_bridge`: {"0": 27, "1": 30, "2": 31, "3": 35, "4": 41, "5": 28}
- `monolithic_bridge`: {"0": 27, "1": 30, "2": 31, "3": 35, "4": 41, "5": 28}
- `hub_dictionary_only`: {"0": 27, "1": 30, "2": 31, "3": 35, "4": 41, "5": 28}
- `hub_feature_router`: {"0": 28, "1": 41, "2": 29, "3": 30, "4": 34, "5": 30}
- `hub_sticky_router`: {"0": 25, "1": 42, "2": 30, "3": 30, "4": 33, "5": 32}
- `hub_sticky_protected_mixed_bit_frontier`: {"0": 25, "1": 42, "2": 30, "3": 30, "4": 33, "5": 32}
- `hub_sticky_frontier_verifier_stop`: {"0": 25, "1": 42, "2": 30, "3": 30, "4": 33, "5": 32}
- `random_router_control`: {"0": 32, "1": 29, "2": 35, "3": 29, "4": 39, "5": 28}
- `confidence_router_control`: {"0": 27, "1": 30, "2": 31, "3": 35, "4": 41, "5": 28}
- `oracle_router_control`: {"0": 30, "1": 25, "2": 35, "3": 43, "4": 35, "5": 24}

## Stop Reasons

- `raw_pairwise_bridge`: {"direct_bridge": 192}
- `monolithic_bridge`: {"direct_bridge": 192}
- `hub_dictionary_only`: {"low_bit_only": 192}
- `hub_feature_router`: {"low_bit_only": 192}
- `hub_sticky_router`: {"low_bit_only": 192}
- `hub_sticky_protected_mixed_bit_frontier`: {"frontier_fixed": 192}
- `hub_sticky_frontier_verifier_stop`: {"confidence_reached": 8, "max_steps": 126, "verifier_harm": 58}
- `random_router_control`: {"low_bit_only": 192}
- `confidence_router_control`: {"low_bit_only": 192}
- `oracle_router_control`: {"low_bit_only": 192}

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
