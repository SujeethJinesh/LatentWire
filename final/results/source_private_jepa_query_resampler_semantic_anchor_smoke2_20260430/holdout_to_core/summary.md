# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `False`
- train/eval families: `holdout -> core`
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
- JEPA query entropy: `1.3269524680652056`
- JEPA context variance: `0.024336220265017246`
- receiver effective rank: `32`
- min decision score: `0.3`
- exact eval surface overlap count: `152`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.125 |
| 8 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.047 | 1.000 | 0.250 |
