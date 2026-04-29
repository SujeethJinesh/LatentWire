# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- candidate atom view: `synonym_stress`
- candidate calibration: `all_public`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.875 | 0.250 | 0.266 | 0.625 | 0.566 | 1.000 | 1.000 |
| 8 | `True` | 0.875 | 0.250 | 0.270 | 0.625 | 0.562 | 1.000 | 1.000 |
