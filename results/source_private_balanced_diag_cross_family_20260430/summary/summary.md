# Source-Private Balanced Diagnostic Packet Gate

- pass gate: `True`
- runs: `4`
- budget bytes: `2`
- min packet accuracy: `1.000`
- max public-only accuracy: `0.178`
- min packet-public CI95 low: `0.788`
- max public-target CI95 high: `-0.022`

| Direct run | Public-only run | families | n | packet | public | target | best control | packet-public CI | public-target CI | parity |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| `results/source_private_balanced_diag_cross_family_20260430/direct_holdout_n500_seed29` | `results/source_private_balanced_diag_cross_family_20260430/public_core_to_holdout_n500_seed29` | core->holdout | 500 | 1.000 | 0.178 | 0.250 | 0.250 | [0.788, 0.856] | [-0.118, -0.022] | `True` |
| `results/source_private_balanced_diag_cross_family_20260430/direct_holdout_n500_seed31` | `results/source_private_balanced_diag_cross_family_20260430/public_core_to_holdout_n500_seed31` | core->holdout | 500 | 1.000 | 0.142 | 0.250 | 0.250 | [0.828, 0.888] | [-0.156, -0.060] | `True` |
| `results/source_private_balanced_diag_cross_family_20260430/direct_core_n500_seed29` | `results/source_private_balanced_diag_cross_family_20260430/public_holdout_to_core_n500_seed29` | holdout->core | 500 | 1.000 | 0.178 | 0.250 | 0.250 | [0.788, 0.856] | [-0.118, -0.022] | `True` |
| `results/source_private_balanced_diag_cross_family_20260430/direct_core_n500_seed31` | `results/source_private_balanced_diag_cross_family_20260430/public_holdout_to_core_n500_seed31` | holdout->core | 500 | 1.000 | 0.142 | 0.250 | 0.250 | [0.828, 0.888] | [-0.156, -0.060] | `True` |

Budget-2 direct diagnostic packet must pass strict controls; public-only diag receiver must have CI95 high <= target+0.05; packet-public CI95 low must be >= 0.10; eval IDs/families/answers must match exactly; public train/eval IDs must be disjoint; and both runs must use plausible-decoy diag_only config.

Balanced plausible-decoy diagnostic tables remove obvious X-code distractors and public semantic shortcuts. A direct 2-byte private diagnostic packet remains sufficient, while a trained public-only diagnostic receiver does not solve the task.
