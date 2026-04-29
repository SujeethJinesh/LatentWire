# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `holdout:768` / `core:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `False` | `False` | `False` | `True` | `False` | 0.381 | 0.375 | n/a | n/a | 0.492 | 0.391 | 0.250 | 0.250 | n/a | 0.117 | 0.016 |
