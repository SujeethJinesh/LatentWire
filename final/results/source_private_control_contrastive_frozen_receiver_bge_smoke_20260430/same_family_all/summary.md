# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `True`
- train/eval families: `all -> all`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_mid_last_mean`
- receiver mode: `contrastive_bilinear`
- contrastive negative sources: `2`
- min decision score: `0.3`
- exact eval surface overlap count: `44`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.031 | 1.000 | 0.938 |
| 4 | `True` | 0.625 | 0.250 | 0.250 | 0.375 | 0.219 | 1.000 | 1.000 |
