# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `False`
- train/eval families: `all -> all`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- min decision score: `0.7`
- exact eval surface overlap count: `352`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.812 | 0.250 | 0.250 | 0.562 | 0.500 | 1.000 | 0.750 |
| 8 | `False` | 0.938 | 0.250 | 0.250 | 0.688 | 0.633 | 1.000 | 0.750 |
