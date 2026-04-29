# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- candidate atom view: `native`
- candidate calibration: `all_public`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 1.000 | 0.250 | 0.270 | 0.750 | 0.699 | 1.000 | 1.000 |
| 8 | `True` | 1.000 | 0.250 | 0.273 | 0.750 | 0.695 | 1.000 | 1.000 |
