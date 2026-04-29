# Anchor-Relative Sparse Packet Gate

- pass gate: `False`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': True}`
- budgets: `[2, 4, 6, 8]`
- sparse accuracy range: `0.125-0.496`
- max sparse-target delta: `0.246`
- min sparse-control delta: `-0.211`

## Rows

| Direction | Budget | N | Sparse | Target | Best control | Sparse-target | Sparse-control | Controls ok | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 2 | 512 | 0.242 | 0.250 | 0.453 | -0.008 | -0.211 | `False` | `False` |
| core_to_holdout | 4 | 512 | 0.250 | 0.250 | 0.271 | 0.000 | -0.021 | `True` | `False` |
| core_to_holdout | 6 | 512 | 0.125 | 0.250 | 0.283 | -0.125 | -0.158 | `False` | `False` |
| core_to_holdout | 8 | 512 | 0.250 | 0.250 | 0.375 | 0.000 | -0.125 | `False` | `False` |
| holdout_to_core | 2 | 512 | 0.496 | 0.250 | 0.250 | 0.246 | 0.246 | `True` | `True` |
| holdout_to_core | 4 | 512 | 0.248 | 0.250 | 0.250 | -0.002 | -0.002 | `True` | `False` |
| holdout_to_core | 6 | 512 | 0.270 | 0.250 | 0.250 | 0.020 | 0.020 | `True` | `False` |
| holdout_to_core | 8 | 512 | 0.373 | 0.250 | 0.262 | 0.123 | 0.111 | `True` | `True` |

Pass rule: At least one budget per direction must beat target by >=0.10, beat the best source-destroying control by >=0.05, and keep every control within target+0.03.
