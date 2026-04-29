# Source-Private Coded-Label Risk Gate

- pass gate: `True`
- examples per seed: `160`
- seeds: `[29, 31, 37]`
- budget bytes: `2`

| Seed | Transform | Pass | Hash gate | Matched | Target | Source controls | Reviewer negatives | Oracles |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 29 | `baseline` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 29 | `label_rename` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 29 | `diagnostic_code_remap` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 29 | `candidate_pool_permutation` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 29 | `label_code_order_composed` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 31 | `baseline` | `True` | `True` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 31 | `label_rename` | `True` | `True` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 31 | `diagnostic_code_remap` | `True` | `True` | 1.000 | 0.250 | 0.263 | 0.250 | 1.000 |
| 31 | `candidate_pool_permutation` | `True` | `True` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 31 | `label_code_order_composed` | `True` | `True` | 1.000 | 0.250 | 0.263 | 0.250 | 1.000 |
| 37 | `baseline` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 37 | `label_rename` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 37 | `diagnostic_code_remap` | `True` | `True` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 37 | `candidate_pool_permutation` | `True` | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 37 | `label_code_order_composed` | `True` | `True` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |

## Transform Summary

| Transform | Pass | Min matched | Max target | Max source control | Max reviewer negative |
|---|---|---:|---:|---:|---:|
| `baseline` | `True` | 1.000 | 0.250 | 0.256 | 0.250 |
| `label_rename` | `True` | 1.000 | 0.250 | 0.256 | 0.250 |
| `diagnostic_code_remap` | `True` | 1.000 | 0.250 | 0.263 | 0.250 |
| `candidate_pool_permutation` | `True` | 1.000 | 0.250 | 0.256 | 0.250 |
| `label_code_order_composed` | `True` | 1.000 | 0.250 | 0.263 | 0.250 |

Pass rule: For every seed and transform, exact IDs must stay fixed; the intended surface hash must change; matched 2-byte packet accuracy must be >=0.95; source-destroying and reviewer-negative controls must remain within +0.03 of target-only; and positive oracles must not trail the packet.
