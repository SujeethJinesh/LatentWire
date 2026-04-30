# Learned Synonym Dictionary Direction

- direction: `core_to_holdout`
- pass gate: `False`
- train/eval families: `core -> holdout`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- exact eval surface overlap count: `96`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.500 | 0.250 | 0.375 | 0.250 | 0.195 | 1.000 | 0.375 |
