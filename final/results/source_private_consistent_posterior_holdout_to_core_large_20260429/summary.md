# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `holdout:1536` / `core:1024`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `False` | `False` | `False` | `True` | `True` | 0.365 | 0.368 | n/a | n/a | 0.502 | 0.495 | 0.250 | 0.250 | n/a | 0.134 | 0.127 |
