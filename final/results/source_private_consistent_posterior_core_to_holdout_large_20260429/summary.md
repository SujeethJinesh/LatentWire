# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `core:1536` / `holdout:1024`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `False` | `False` | `False` | `False` | `False` | 0.253 | 0.370 | n/a | n/a | 0.146 | 0.354 | 0.250 | 0.250 | n/a | -0.225 | -0.017 |
