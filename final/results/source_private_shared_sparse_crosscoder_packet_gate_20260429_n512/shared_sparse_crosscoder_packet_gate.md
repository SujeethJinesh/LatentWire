# Shared Sparse Crosscoder Packet Gate

- pass gate: `True`
- direction pass: `{'core_to_holdout': True, 'holdout_to_core': True, 'same_family_all': True}`
- cross-family pass: `True`
- budgets: `[4, 8]`
- candidate atom view: `native`
- max shared sparse accuracy: `1.000`
- max shared-target delta: `0.750`

## Rows

| Direction | Budget | N | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 4 | 512 | `True` | 1.000 | 0.250 | 0.250 | 0.750 | 0.711 | 1.000 |
| core_to_holdout | 8 | 512 | `True` | 1.000 | 0.250 | 0.256 | 0.750 | 0.713 | 1.000 |
| holdout_to_core | 4 | 512 | `False` | 0.875 | 0.250 | 0.250 | 0.625 | 0.580 | 1.000 |
| holdout_to_core | 8 | 512 | `True` | 0.875 | 0.250 | 0.252 | 0.625 | 0.582 | 1.000 |
| same_family_all | 4 | 512 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.645 | 1.000 |
| same_family_all | 8 | 512 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.648 | 1.000 |

Pass rule: Bidirectional cross-family pass requires at least one budget per direction with matched shared sparse packet beating target by >=0.10, best source-destroying control by >=0.05, all controls within target+0.03, paired CI95 lower bound >0, oracle >=0.90, and top-atom knockout removing >=50% of lift.
