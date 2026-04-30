# Learned Synonym Dictionary Packet Gate

- pass gate: `False`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}`
- cross-family pass: `False`
- budgets: `[4]`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_last_mean`
- min decision score: `0.7`
- max learned packet accuracy: `0.625`
- max learned-target delta: `0.375`

## Rows

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 4 | 32 | `False` | 0.500 | 0.250 | 0.250 | 0.250 | 0.031 | 1.000 |
| holdout_to_core | 4 | 32 | `False` | 0.625 | 0.250 | 0.250 | 0.375 | 0.219 | 1.000 |
| same_family_all | 4 | 32 | `False` | 0.562 | 0.250 | 0.250 | 0.312 | 0.125 | 1.000 |

Pass rule: Bidirectional cross-family pass requires at least one budget per direction with learned synonym dictionary packet beating target by >=0.15, best source-destroying control by >=0.10, all source-destroying controls within target+0.03, paired CI95 lower bound >0.05, learned candidate oracle >=0.80, and top-feature knockout removing >=50% of lift.
