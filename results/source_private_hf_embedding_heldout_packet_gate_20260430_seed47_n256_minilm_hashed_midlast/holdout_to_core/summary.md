# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `False`
- train/eval families: `holdout -> core`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hashed_hf_mid_last_mean`
- receiver mode: `atom_ridge`
- contrastive negative sources: `0`
- contrastive rank: `None`
- low-rank factor epochs: `None`
- low-rank factor lr: `None`
- low-rank factor loss: `None`
- low-rank factor seed: `None`
- JEPA query count: `None`
- JEPA hidden dim: `None`
- JEPA trainable factors: `None`
- JEPA train epochs: `None`
- JEPA lr: `None`
- JEPA weight decay: `None`
- JEPA query entropy: `None`
- JEPA context variance: `None`
- receiver effective rank: `None`
- min decision score: `0.5`
- exact eval surface overlap count: `608`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.625 | 0.250 | 0.250 | 0.375 | 0.289 | 1.000 | 0.750 |
| 8 | `False` | 0.625 | 0.250 | 0.375 | 0.375 | 0.281 | 1.000 | 0.750 |
