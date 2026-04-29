# Learned Synonym Dictionary Direction

- direction: `core_to_holdout`
- pass gate: `True`
- train/eval families: `core -> holdout`
- candidate atom view: `native`
- candidate calibration: `all_public`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 1.000 | 0.250 | 0.250 | 0.750 | 0.699 | 1.000 | 1.000 |
| 8 | `False` | 1.000 | 0.250 | 0.375 | 0.750 | 0.695 | 1.000 | 1.000 |
