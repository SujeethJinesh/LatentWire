# Source-Private Codebook-Remap Gate

- pass gate: `True`
- examples: `500`
- family set: `all`
- seeds: `[29, 31, 37]`
- budgets: `[2, 4, 8, 16]`
- exact ID parity across seeds: `True`
- public surface parity across seeds: `True`
- unique codebooks: `3`

| Seed | Budget | Pass | Matched | No-source | Source controls | Reviewer negatives | Oracles | JSON | Free text | Diag masked |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 2 | `True` | 1.000 | 0.250 | 0.254 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 29 | 4 | `True` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 29 | 8 | `True` | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 29 | 16 | `True` | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 31 | 2 | `True` | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 31 | 4 | `True` | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 31 | 8 | `True` | 1.000 | 0.250 | 0.254 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 31 | 16 | `True` | 1.000 | 0.250 | 0.254 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 37 | 2 | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 37 | 4 | `True` | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 37 | 8 | `True` | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |
| 37 | 16 | `True` | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 | 0.250 | 0.250 | 0.250 |

Pass rule: Every seed/budget must pass the hidden-repair packet gate; exact IDs and public candidate labels must remain identical across seeds; diagnostic codebooks must differ.
