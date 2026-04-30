# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `True`
- train/eval families: `all -> all`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- min decision score: `0.7`
- exact eval surface overlap count: `704`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.812 | 0.250 | 0.250 | 0.562 | 0.518 | 1.000 | 0.938 |
| 8 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.646 | 1.000 | 0.938 |
