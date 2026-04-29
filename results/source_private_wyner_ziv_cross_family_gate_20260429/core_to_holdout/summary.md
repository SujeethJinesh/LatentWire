# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `False` | `False` | `False` | `False` | `False` | 0.236 | 0.127 | 0.252 | n/a | 0.125 | n/a | 0.250 | 0.250 | n/a | -0.002 | n/a |
| 4 | `False` | `False` | `False` | `False` | `False` | `False` | 0.246 | 0.174 | 0.131 | n/a | 0.207 | n/a | 0.250 | 0.252 | n/a | 0.033 | n/a |
| 6 | `False` | `False` | `False` | `False` | `False` | `False` | 0.240 | 0.146 | 0.131 | n/a | 0.207 | n/a | 0.250 | 0.250 | n/a | 0.061 | n/a |
