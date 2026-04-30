# Learned Synonym Dictionary Direction

- direction: `core_to_holdout`
- pass gate: `False`
- train/eval families: `core -> holdout`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- receiver mode: `jepa_query_resampler`
- contrastive negative sources: `2`
- contrastive rank: `None`
- low-rank factor epochs: `None`
- low-rank factor lr: `None`
- low-rank factor loss: `None`
- low-rank factor seed: `None`
- JEPA query count: `8`
- JEPA hidden dim: `16`
- JEPA query entropy: `1.3317790863635186`
- JEPA context variance: `0.006072958243700155`
- receiver effective rank: `128`
- min decision score: `0.2`
- exact eval surface overlap count: `24`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.500 | 0.250 | 0.250 | 0.250 | 0.141 | 1.000 | 0.500 |
| 8 | `False` | 0.500 | 0.250 | 0.250 | 0.250 | 0.141 | 1.000 | 0.750 |
