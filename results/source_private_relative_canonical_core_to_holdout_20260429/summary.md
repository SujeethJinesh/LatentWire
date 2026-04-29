# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Target | Best no-source | Relative - scalar | Canonical - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `False` | `False` | `False` | `False` | 0.246 | 0.225 | n/a | n/a | 0.207 | 0.250 | 0.250 | n/a | -0.018 |
