# Shared Sparse Crosscoder Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- exact ID parity: `True`

| Budget | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.875 | 0.250 | 0.250 | 0.625 | 0.580 | 1.000 | 0.875 |
| 8 | `True` | 0.875 | 0.250 | 0.252 | 0.625 | 0.582 | 1.000 | 1.000 |
