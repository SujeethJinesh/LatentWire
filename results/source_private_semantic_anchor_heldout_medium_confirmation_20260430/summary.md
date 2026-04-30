# Semantic-Anchor Held-Out Medium Confirmation

- pass gate: `True`
- seeds: `[47, 53, 59]`
- rows passed: `18/18`
- min CI95 low vs target: `0.457`
- min learned-target lift: `0.500`
- max best-control accuracy: `0.254`
- min oracle accuracy: `0.875`
- exact transformed held-out overlap zero: `True`

| Seed | Direction | Budget | Pass | Learned | Target | Best control | Delta target | CI95 low | Oracle |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 47 | core_to_holdout | 4 | `True` | 0.750 | 0.250 | 0.250 | 0.500 | 0.457 | 1.000 |
| 47 | core_to_holdout | 8 | `True` | 1.000 | 0.250 | 0.252 | 0.750 | 0.709 | 0.875 |
| 47 | holdout_to_core | 4 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.582 | 0.875 |
| 47 | holdout_to_core | 8 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.582 | 1.000 |
| 47 | same_family_all | 4 | `True` | 0.812 | 0.250 | 0.250 | 0.562 | 0.520 | 0.938 |
| 47 | same_family_all | 8 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.645 | 0.938 |
| 53 | core_to_holdout | 4 | `True` | 0.750 | 0.250 | 0.250 | 0.500 | 0.457 | 1.000 |
| 53 | core_to_holdout | 8 | `True` | 1.000 | 0.250 | 0.250 | 0.750 | 0.713 | 0.875 |
| 53 | holdout_to_core | 4 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.582 | 0.875 |
| 53 | holdout_to_core | 8 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.582 | 1.000 |
| 53 | same_family_all | 4 | `True` | 0.812 | 0.250 | 0.250 | 0.562 | 0.518 | 0.938 |
| 53 | same_family_all | 8 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.646 | 0.938 |
| 59 | core_to_holdout | 4 | `True` | 0.750 | 0.250 | 0.250 | 0.500 | 0.457 | 1.000 |
| 59 | core_to_holdout | 8 | `True` | 1.000 | 0.250 | 0.254 | 0.750 | 0.715 | 0.875 |
| 59 | holdout_to_core | 4 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.580 | 0.875 |
| 59 | holdout_to_core | 8 | `True` | 0.875 | 0.250 | 0.252 | 0.625 | 0.584 | 1.000 |
| 59 | same_family_all | 4 | `True` | 0.812 | 0.250 | 0.250 | 0.562 | 0.518 | 0.938 |
| 59 | same_family_all | 8 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.648 | 0.938 |

Pass rule: All three seeds must pass bidirectional cross-family with at least one passing row in each direction; here every row passes and exact transformed held-out overlap remains zero.
