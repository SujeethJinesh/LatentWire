# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `False`
- train/eval families: `holdout -> core`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- min decision score: `0.7`
- exact eval surface overlap count: `608`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.875 | 0.250 | 0.250 | 0.625 | 0.562 | 1.000 | 0.625 |
| 8 | `False` | 0.875 | 0.250 | 0.250 | 0.625 | 0.562 | 1.000 | 0.750 |
