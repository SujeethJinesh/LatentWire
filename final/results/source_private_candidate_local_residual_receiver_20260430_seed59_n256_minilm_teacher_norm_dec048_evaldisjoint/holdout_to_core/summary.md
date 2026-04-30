# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public_eval_disjoint`
- text feature mode: `hf_last_mean`
- adapter target mode: `semantic_anchor_teacher`
- decoder score mode: `candidate_local_residual_norm`
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
- min decision score: `0.48`
- calibration/eval exact ID overlap count: `0`
- exact eval surface overlap count: `608`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | 0.250 | 0.250 | 0.258 | 0.000 | 0.000 | 0.000 | 0.625 |
| 4 | `True` | 0.500 | 0.250 | 0.258 | 0.250 | 0.199 | 1.000 | 1.000 |
| 8 | `True` | 0.500 | 0.250 | 0.254 | 0.250 | 0.195 | 1.000 | 1.000 |
