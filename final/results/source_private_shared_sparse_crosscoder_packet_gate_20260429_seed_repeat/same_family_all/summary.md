# Shared Sparse Crosscoder Direction

- direction: `same_family_all`
- pass gate: `True`
- train/eval families: `all -> all`
- exact ID parity: `True`

| Budget | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.609 | 1.000 | 0.938 |
| 8 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.609 | 1.000 | 1.000 |
