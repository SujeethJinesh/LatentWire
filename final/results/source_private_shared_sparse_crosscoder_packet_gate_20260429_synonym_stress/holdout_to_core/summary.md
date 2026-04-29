# Shared Sparse Crosscoder Direction

- direction: `holdout_to_core`
- pass gate: `False`
- train/eval families: `holdout -> core`
- exact ID parity: `True`

| Budget | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.250 |
| 8 | `False` | 0.125 | 0.250 | 0.250 | -0.125 | -0.188 | 0.000 | 0.250 |
