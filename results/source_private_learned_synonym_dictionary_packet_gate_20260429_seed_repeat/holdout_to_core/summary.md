# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- candidate atom view: `synonym_stress`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- exact eval surface overlap count: `1024`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.875 | 0.250 | 0.258 | 0.625 | 0.562 | 1.000 | 1.000 |
