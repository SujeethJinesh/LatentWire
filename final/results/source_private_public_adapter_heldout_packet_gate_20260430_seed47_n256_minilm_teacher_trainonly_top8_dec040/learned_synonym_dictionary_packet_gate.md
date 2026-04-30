# Learned Synonym Dictionary Packet Gate

- pass gate: `False`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}`
- cross-family pass: `False`
- budgets: `[2, 4, 8]`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `train_only`
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
- min decision score: `0.4`
- max learned packet accuracy: `0.875`
- max learned-target delta: `0.625`

## Rows

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 2 | 256 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | -0.062 | 0.000 |
| core_to_holdout | 4 | 256 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.055 | 1.000 |
| core_to_holdout | 8 | 256 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.031 | 1.000 |
| holdout_to_core | 2 | 256 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| holdout_to_core | 4 | 256 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.086 | 1.000 |
| holdout_to_core | 8 | 256 | `False` | 0.375 | 0.250 | 0.375 | 0.125 | 0.051 | 1.000 |
| same_family_all | 2 | 256 | `False` | 0.562 | 0.250 | 0.312 | 0.312 | 0.258 | 1.000 |
| same_family_all | 4 | 256 | `False` | 0.875 | 0.250 | 0.312 | 0.625 | 0.562 | 1.000 |
| same_family_all | 8 | 256 | `False` | 0.750 | 0.250 | 0.312 | 0.500 | 0.426 | 1.000 |

Pass rule: Bidirectional cross-family pass requires at least one budget per direction with learned synonym dictionary packet beating target by >=0.15, best source-destroying control by >=0.10, all source-destroying controls within target+0.03, paired CI95 lower bound >0.05, learned candidate oracle >=0.80, and top-feature knockout removing >=50% of lift.
