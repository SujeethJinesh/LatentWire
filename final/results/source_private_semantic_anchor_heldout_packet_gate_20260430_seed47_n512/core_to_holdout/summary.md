# Learned Synonym Dictionary Direction

- direction: `core_to_holdout`
- pass gate: `True`
- train/eval families: `core -> holdout`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- min decision score: `0.7`
- exact eval surface overlap count: `192`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.750 | 0.250 | 0.250 | 0.500 | 0.457 | 1.000 | 1.000 |
| 8 | `True` | 1.000 | 0.250 | 0.252 | 0.750 | 0.709 | 1.000 | 0.875 |
