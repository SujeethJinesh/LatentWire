# Shared Sparse Crosscoder Direction

- direction: `core_to_holdout`
- pass gate: `True`
- train/eval families: `core -> holdout`
- exact ID parity: `True`

| Budget | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 1.000 | 0.250 | 0.250 | 0.750 | 0.711 | 1.000 | 1.000 |
| 8 | `True` | 1.000 | 0.250 | 0.256 | 0.750 | 0.713 | 1.000 | 1.000 |
