# Learned Synonym Dictionary Packet Gate

- pass gate: `False`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}`
- cross-family pass: `False`
- budgets: `[2, 4]`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_mid_last_mean`
- receiver mode: `contrastive_low_rank_factor`
- contrastive negative sources: `2`
- contrastive rank: `8`
- low-rank factor epochs: `220`
- low-rank factor lr: `0.02`
- low-rank factor loss: `bce`
- low-rank factor seed: `947`
- min decision score: `0.3`
- max learned packet accuracy: `0.250`
- max learned-target delta: `0.000`

## Rows

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 2 | 32 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| core_to_holdout | 4 | 32 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| holdout_to_core | 2 | 32 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| holdout_to_core | 4 | 32 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| same_family_all | 2 | 32 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| same_family_all | 4 | 32 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |

Pass rule: Bidirectional cross-family pass requires at least one budget per direction with learned synonym dictionary packet beating target by >=0.15, best source-destroying control by >=0.10, all source-destroying controls within target+0.03, paired CI95 lower bound >0.05, learned candidate oracle >=0.80, and top-feature knockout removing >=50% of lift.
