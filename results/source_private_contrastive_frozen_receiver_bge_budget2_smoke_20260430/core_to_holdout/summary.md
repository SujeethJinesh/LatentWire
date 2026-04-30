# Learned Synonym Dictionary Direction

- direction: `core_to_holdout`
- pass gate: `True`
- train/eval families: `core -> holdout`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_mid_last_mean`
- receiver mode: `contrastive_bilinear`
- min decision score: `0.3`
- exact eval surface overlap count: `12`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | 0.625 | 0.250 | 0.375 | 0.375 | 0.219 | 1.000 | 0.875 |
| 4 | `True` | 0.750 | 0.250 | 0.250 | 0.500 | 0.344 | 1.000 | 0.875 |
