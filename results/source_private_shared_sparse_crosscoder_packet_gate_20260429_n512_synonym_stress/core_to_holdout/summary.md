# Shared Sparse Crosscoder Direction

- direction: `core_to_holdout`
- pass gate: `False`
- train/eval families: `core -> holdout`
- exact ID parity: `True`

| Budget | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.098 | 1.000 | 0.375 |
| 8 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.098 | 1.000 | 0.375 |
