# Source-Private Tool-Trace Learned Syndrome

- pass gate: `True`
- train/eval: `all:512` / `all:256`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `False` | 0.500 | 0.250 | 0.379 | 0.121 | 0.633 |
| 2 | `False` | 0.738 | 0.250 | 0.410 | 0.328 | 0.922 |
| 4 | `False` | 0.895 | 0.250 | 0.352 | 0.543 | 1.000 |
| 6 | `True` | 0.945 | 0.250 | 0.285 | 0.660 | 1.000 |
| 8 | `True` | 0.980 | 0.250 | 0.250 | 0.730 | 1.000 |
| 12 | `False` | 0.996 | 0.250 | 0.277 | 0.719 | 1.000 |
