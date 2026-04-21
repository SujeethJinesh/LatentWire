# Toy Hub Router / Frontier Sweep

Deterministic route-conditioned sweep for hub decoding, protected frontiers, and verifier stopping.

| Method | Router | Frontier | Stop | Accuracy | Route acc | Stability | Atom recovery | Avg stop steps | Over-refine | Frontier delta | Stop delta | Bytes proxy | Compute proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_pairwise_bridge | pairwise_control | False | False | 0.7344 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 50400.0000 | 510.0000 |
| conditional_prior_base | conditional_prior | False | False | 0.6250 | 0.6979 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 22914.0000 | 462.0000 |
| conditional_prior_frontier | conditional_prior | True | False | 0.6354 | 0.6979 | 1.0000 | 0.5286 | 1.0000 | 0.0000 | 0.0104 | 0.0000 | 22964.0000 | 563.0000 |
| conditional_prior_frontier_stop | conditional_prior | True | True | 0.6198 | 0.6979 | 1.0000 | 0.5286 | 2.8438 | 0.4531 | 0.0104 | -0.0156 | 22964.0000 | 650.3125 |
| feature_router_base | feature_router | False | False | 0.6094 | 0.6302 | 0.9010 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 22914.0000 | 462.0000 |
| feature_router_frontier | feature_router | True | False | 0.6042 | 0.6302 | 0.9010 | 0.5286 | 1.0000 | 0.0000 | -0.0052 | 0.0000 | 22964.0000 | 563.0000 |
| feature_router_frontier_stop | feature_router | True | True | 0.6042 | 0.6302 | 0.9010 | 0.5286 | 2.9688 | 0.4740 | -0.0052 | 0.0000 | 22964.0000 | 654.0625 |
| sticky_router_base | sticky_router | False | False | 0.6042 | 0.6250 | 0.9219 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 22914.0000 | 462.0000 |
| sticky_router_frontier | sticky_router | True | False | 0.5990 | 0.6250 | 0.9219 | 0.5286 | 1.0000 | 0.0000 | -0.0052 | 0.0000 | 22964.0000 | 563.0000 |
| sticky_router_frontier_stop | sticky_router | True | True | 0.5938 | 0.6250 | 0.9219 | 0.5286 | 2.9688 | 0.4583 | -0.0052 | -0.0052 | 22964.0000 | 654.0625 |
| confidence_router_base | confidence_router | False | False | 0.6250 | 0.6979 | 0.9896 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 22914.0000 | 462.0000 |
| confidence_router_frontier | confidence_router | True | False | 0.6354 | 0.6979 | 0.9896 | 0.5286 | 1.0000 | 0.0000 | 0.0104 | 0.0000 | 22964.0000 | 563.0000 |
| confidence_router_frontier_stop | confidence_router | True | True | 0.6198 | 0.6979 | 0.9896 | 0.5286 | 2.8438 | 0.4531 | 0.0104 | -0.0156 | 22964.0000 | 650.3125 |
| random_router_base | random_router | False | False | 0.3177 | 0.1875 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 22914.0000 | 462.0000 |
| random_router_frontier | random_router | True | False | 0.3125 | 0.1875 | 1.0000 | 0.5286 | 1.0000 | 0.0000 | -0.0052 | 0.0000 | 22964.0000 | 563.0000 |
| random_router_frontier_stop | random_router | True | True | 0.2969 | 0.1875 | 1.0000 | 0.5286 | 2.9375 | 0.2031 | -0.0052 | -0.0156 | 22964.0000 | 653.1250 |
| oracle_router_base | oracle_router | False | False | 0.8229 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 22914.0000 | 462.0000 |
| oracle_router_frontier | oracle_router | True | False | 0.8125 | 1.0000 | 1.0000 | 0.5286 | 1.0000 | 0.0000 | -0.0104 | 0.0000 | 22964.0000 | 563.0000 |
| oracle_router_frontier_stop | oracle_router | True | True | 0.8073 | 1.0000 | 1.0000 | 0.5286 | 2.7812 | 0.5938 | -0.0104 | -0.0052 | 22964.0000 | 648.4375 |

## Router Summary

| Router | Route acc | Stability | Base acc | Frontier acc | Stop acc | Frontier gain | Stop gain | Frontier atom recovery | Stop over-refine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conditional_prior | 0.6979 | 1.0000 | 0.6250 | 0.6354 | 0.6198 | 0.0104 | -0.0156 | 0.5286 | 0.4531 |
| feature_router | 0.6302 | 0.9010 | 0.6094 | 0.6042 | 0.6042 | -0.0052 | 0.0000 | 0.5286 | 0.4740 |
| sticky_router | 0.6250 | 0.9219 | 0.6042 | 0.5990 | 0.5938 | -0.0052 | -0.0052 | 0.5286 | 0.4583 |
| confidence_router | 0.6979 | 0.9896 | 0.6250 | 0.6354 | 0.6198 | 0.0104 | -0.0156 | 0.5286 | 0.4531 |
| random_router | 0.1875 | 1.0000 | 0.3177 | 0.3125 | 0.2969 | -0.0052 | -0.0156 | 0.5286 | 0.2031 |
| oracle_router | 1.0000 | 1.0000 | 0.8229 | 0.8125 | 0.8073 | -0.0104 | -0.0052 | 0.5286 | 0.5938 |

## Interpretation

This sweep separates the hub base from later frontier and stop heuristics. The best non-oracle hub base is conditional_prior at 0.6250, while oracle routing raises the same hub base to 0.8229, above raw pairwise 0.7344. However, the current frontier never adds more than 0.0104 accuracy and the stop heuristic never adds more than 0.0000. That means route assignment is a real headroom source, but the current quant-error frontier and verifier stop rules are themselves mis-specified and should not be stacked unchanged.

## Sources Consulted

- https://arxiv.org/abs/2502.03714
- https://arxiv.org/abs/2506.14038
- https://arxiv.org/abs/2311.14125
- https://arxiv.org/abs/2204.08396
