# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `holdout:768` / `core:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `False` | `False` | `False` | `False` | `False` | 0.129 | 0.328 | 0.381 | n/a | 0.375 | n/a | 0.250 | 0.250 | n/a | 0.047 | n/a |
| 4 | `False` | `False` | `True` | `False` | `True` | `False` | 0.139 | 0.338 | 0.414 | n/a | 0.498 | n/a | 0.250 | 0.250 | n/a | 0.160 | n/a |
| 6 | `False` | `True` | `True` | `False` | `True` | `False` | 0.176 | 0.623 | 0.564 | n/a | 0.498 | n/a | 0.250 | 0.250 | n/a | -0.125 | n/a |
