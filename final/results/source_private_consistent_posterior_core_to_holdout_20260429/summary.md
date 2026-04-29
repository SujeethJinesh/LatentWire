# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `False` | `False` | `False` | `False` | `False` | 0.246 | 0.225 | n/a | n/a | 0.207 | 0.381 | 0.250 | 0.266 | n/a | -0.018 | 0.156 |
