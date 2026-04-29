# Source-Private Learned Syndrome Smoke

- pass gate: `True`
- train/eval: `512/256`
- candidates: `4`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full text |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `True` | 0.797 | 0.250 | 0.281 | 0.516 | 1.000 |
| 2 | `True` | 0.902 | 0.250 | 0.266 | 0.637 | 1.000 |
| 4 | `False` | 0.988 | 0.250 | 0.305 | 0.684 | 1.000 |
| 8 | `False` | 1.000 | 0.250 | 0.309 | 0.691 | 1.000 |
