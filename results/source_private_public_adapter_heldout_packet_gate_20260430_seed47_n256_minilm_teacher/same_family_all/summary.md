# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `False`
- train/eval families: `all -> all`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_last_mean`
- adapter target mode: `semantic_anchor_teacher`
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
- min decision score: `0.2`
- calibration/eval exact ID overlap count: `256`
- exact eval surface overlap count: `352`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | 0.750 | 0.250 | 0.254 | 0.500 | 0.441 | 1.000 | 0.625 |
| 4 | `False` | 0.875 | 0.250 | 0.312 | 0.625 | 0.562 | 1.000 | 0.875 |
| 8 | `False` | 0.750 | 0.250 | 0.375 | 0.500 | 0.426 | 1.000 | 0.875 |
