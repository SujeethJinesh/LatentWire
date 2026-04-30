# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- receiver mode: `contrastive_bilinear`
- min decision score: `0.3`
- exact eval surface overlap count: `76`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 1.000 | 0.250 | 0.250 | 0.750 | 0.594 | 1.000 | 1.000 |
| 8 | `False` | 1.000 | 0.250 | 0.375 | 0.750 | 0.594 | 1.000 | 1.000 |
