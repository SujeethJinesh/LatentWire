# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Syndrome | Scalar | QJL | Relative | Target | Best no-source | Relative - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `False` | `False` | `False` | 0.246 | 0.225 | n/a | 0.207 | 0.250 | 0.250 | -0.018 |
