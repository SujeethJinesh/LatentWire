# Learned Synonym Dictionary Packet Gate

- pass gate: `True`
- direction pass: `{'core_to_holdout': True, 'holdout_to_core': True, 'same_family_all': True}`
- cross-family pass: `True`
- budgets: `[2, 4, 8]`
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
- min decision score: `0.48`
- max learned packet accuracy: `0.625`
- max learned-target delta: `0.375`

## Rows

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 2 | 512 | `False` | 0.375 | 0.250 | 0.375 | 0.125 | 0.096 | 1.000 |
| core_to_holdout | 4 | 512 | `False` | 0.625 | 0.250 | 0.375 | 0.375 | 0.332 | 1.000 |
| core_to_holdout | 8 | 512 | `True` | 0.625 | 0.250 | 0.250 | 0.375 | 0.334 | 1.000 |
| holdout_to_core | 2 | 512 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| holdout_to_core | 4 | 512 | `False` | 0.375 | 0.250 | 0.256 | 0.125 | 0.098 | 1.000 |
| holdout_to_core | 8 | 512 | `True` | 0.500 | 0.250 | 0.256 | 0.250 | 0.215 | 1.000 |
| same_family_all | 2 | 512 | `False` | 0.312 | 0.250 | 0.312 | 0.062 | 0.043 | 1.000 |
| same_family_all | 4 | 512 | `False` | 0.500 | 0.250 | 0.312 | 0.250 | 0.217 | 1.000 |
| same_family_all | 8 | 512 | `True` | 0.562 | 0.250 | 0.250 | 0.312 | 0.271 | 1.000 |

Pass rule: Bidirectional cross-family pass requires at least one budget per direction with learned synonym dictionary packet beating target by >=0.15, best source-destroying control by >=0.10, all source-destroying controls within target+0.03, paired CI95 lower bound >0.05, learned candidate oracle >=0.80, and top-feature knockout removing >=50% of lift. Strict controls include shuffled source packets, atom-ID derangement, private-random atom packets, and a permuted-teacher receiver. Private-random single-atom knockout is reported as a packet-fragility diagnostic but is not a hard veto because low-rate 2-4 atom packets are expected to lose lift when a real transmitted atom is removed.
