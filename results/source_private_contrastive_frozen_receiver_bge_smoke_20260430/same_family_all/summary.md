# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `False`
- train/eval families: `all -> all`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_mid_last_mean`
- receiver mode: `contrastive_bilinear`
- min decision score: `0.3`
- exact eval surface overlap count: `44`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.875 | 0.250 | 0.438 | 0.625 | 0.469 | 1.000 | 0.938 |
| 8 | `False` | 0.938 | 0.250 | 0.438 | 0.688 | 0.531 | 1.000 | 0.875 |
