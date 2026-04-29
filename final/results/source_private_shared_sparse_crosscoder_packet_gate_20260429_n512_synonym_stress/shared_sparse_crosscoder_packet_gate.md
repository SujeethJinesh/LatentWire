# Shared Sparse Crosscoder Packet Gate

- pass gate: `False`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}`
- cross-family pass: `False`
- budgets: `[4, 8]`
- candidate atom view: `synonym_stress`
- max shared sparse accuracy: `0.375`
- max shared-target delta: `0.125`

## Rows

| Direction | Budget | N | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 4 | 512 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.098 | 1.000 |
| core_to_holdout | 8 | 512 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.098 | 1.000 |
| holdout_to_core | 4 | 512 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |
| holdout_to_core | 8 | 512 | `False` | 0.125 | 0.250 | 0.250 | -0.125 | -0.154 | 0.000 |
| same_family_all | 4 | 512 | `False` | 0.312 | 0.250 | 0.250 | 0.062 | 0.043 | 1.000 |
| same_family_all | 8 | 512 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | -0.031 | 0.000 |

Pass rule: Bidirectional cross-family pass requires at least one budget per direction with matched shared sparse packet beating target by >=0.10, best source-destroying control by >=0.05, all controls within target+0.03, paired CI95 lower bound >0, oracle >=0.90, and top-atom knockout removing >=50% of lift.
