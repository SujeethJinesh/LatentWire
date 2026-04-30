# Learned Synonym Dictionary Direction

- direction: `core_to_holdout`
- pass gate: `True`
- train/eval families: `core -> holdout`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- exact eval surface overlap count: `96`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.566 | 1.000 | 1.000 |
| 8 | `False` | 1.000 | 0.250 | 0.375 | 0.750 | 0.695 | 1.000 | 0.875 |
