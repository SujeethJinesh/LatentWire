# Conditional Semantic Syndrome Gate

- pass gate: `False`
- candidate view: `synonym_stress`
- direction pass: `{'core_to_holdout': False, 'holdout_to_core': False, 'same_family_all': False}`
- max accuracy: `0.797`
- max lift vs target: `0.547`

| Direction | Budget | Pass | Syndrome | Target | Best control | Delta target | CI95 low | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 2 | `False` | 0.172 | 0.250 | 0.328 | -0.078 | -0.234 | 1.000 |
| core_to_holdout | 4 | `False` | 0.203 | 0.250 | 0.250 | -0.047 | -0.203 | 1.000 |
| core_to_holdout | 8 | `False` | 0.281 | 0.250 | 0.297 | 0.031 | -0.156 | 1.000 |
| holdout_to_core | 2 | `False` | 0.641 | 0.250 | 0.375 | 0.391 | 0.281 | 1.000 |
| holdout_to_core | 4 | `False` | 0.703 | 0.250 | 0.500 | 0.453 | 0.344 | 1.000 |
| holdout_to_core | 8 | `False` | 0.500 | 0.250 | 0.500 | 0.250 | 0.062 | 1.000 |
| same_family_all | 2 | `False` | 0.766 | 0.250 | 0.438 | 0.516 | 0.328 | 1.000 |
| same_family_all | 4 | `False` | 0.797 | 0.250 | 0.438 | 0.547 | 0.344 | 1.000 |
| same_family_all | 8 | `False` | 0.797 | 0.250 | 0.500 | 0.547 | 0.344 | 1.000 |
