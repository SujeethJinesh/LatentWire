# Learned Synonym Dictionary Packet Gate

- pass gate: `False`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}`
- cross-family pass: `False`
- budgets: `[4, 8]`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `semantic_anchor`
- receiver mode: `jepa_query_resampler_control_regularized`
- contrastive negative sources: `2`
- contrastive rank: `None`
- low-rank factor epochs: `None`
- low-rank factor lr: `None`
- low-rank factor loss: `None`
- low-rank factor seed: `None`
- JEPA query count: `8`
- JEPA hidden dim: `16`
- JEPA trainable factors: `True`
- JEPA train epochs: `40`
- JEPA lr: `0.01`
- JEPA weight decay: `0.001`
- min decision score: `0.0`
- max learned packet accuracy: `0.250`
- max learned-target delta: `0.000`

## Rows

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 4 | 64 | `False` | 0.250 | 0.250 | 0.500 | 0.000 | 0.000 | 0.000 |
| core_to_holdout | 8 | 64 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| holdout_to_core | 4 | 64 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| holdout_to_core | 8 | 64 | `False` | 0.250 | 0.250 | 0.266 | 0.000 | 0.000 | 0.000 |
| same_family_all | 4 | 64 | `False` | 0.250 | 0.250 | 0.375 | 0.000 | 0.000 | 0.000 |
| same_family_all | 8 | 64 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |

Pass rule: Bidirectional cross-family pass requires at least one budget per direction with learned synonym dictionary packet beating target by >=0.15, best source-destroying control by >=0.10, all source-destroying controls within target+0.03, paired CI95 lower bound >0.05, learned candidate oracle >=0.80, and top-feature knockout removing >=50% of lift.
