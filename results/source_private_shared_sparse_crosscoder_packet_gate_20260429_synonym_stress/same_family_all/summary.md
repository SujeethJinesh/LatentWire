# Shared Sparse Crosscoder Direction

- direction: `same_family_all`
- pass gate: `False`
- train/eval families: `all -> all`
- exact ID parity: `True`

| Budget | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.312 | 0.250 | 0.250 | 0.062 | 0.023 | 1.000 | 0.312 |
| 8 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | -0.062 | 0.000 | 0.312 |
