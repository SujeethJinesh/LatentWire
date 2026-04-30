# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `False`
- train/eval families: `all -> all`
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
- JEPA query count: `4`
- JEPA hidden dim: `8`
- JEPA query entropy: `1.32733032782705`
- JEPA context variance: `0.027255261921010403`
- receiver effective rank: `32`
- min decision score: `0.3`
- exact eval surface overlap count: `88`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.375 |
| 8 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.375 |
