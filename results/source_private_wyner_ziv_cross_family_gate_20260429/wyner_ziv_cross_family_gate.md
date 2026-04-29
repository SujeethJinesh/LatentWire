# Source-Private Wyner-Ziv Cross-Family Gate

- pass gate: `False`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': False}`
- budgets: `[2, 4, 6]`
- scalar accuracy range: `0.127-0.623`
- minimum scalar-control margin: `-0.496`

## Rows

| Direction | Budget | N | Scalar WZ | Target | Best scalar control | Raw sign | QJL | Canonical RASP | Scalar pass | Canonical pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 2 | 512 | 0.127 | 0.250 | 0.623 | 0.283 | 0.252 | 0.125 | `False` | `False` |
| core_to_holdout | 4 | 512 | 0.174 | 0.250 | 0.529 | 0.365 | 0.131 | 0.207 | `False` | `False` |
| core_to_holdout | 6 | 512 | 0.146 | 0.250 | 0.584 | 0.326 | 0.131 | 0.207 | `False` | `False` |
| holdout_to_core | 2 | 512 | 0.328 | 0.250 | 0.275 | 0.246 | 0.381 | 0.375 | `False` | `False` |
| holdout_to_core | 4 | 512 | 0.338 | 0.250 | 0.250 | 0.129 | 0.414 | 0.498 | `False` | `True` |
| holdout_to_core | 6 | 512 | 0.623 | 0.250 | 0.250 | 0.109 | 0.564 | 0.498 | `True` | `True` |

Pass rule: All 2/4/6-byte rows in both core->holdout and holdout->core must beat target by >=0.15 and best source-destroying scalar control by >=0.15. Otherwise cross-family remains failed/asymmetric.
