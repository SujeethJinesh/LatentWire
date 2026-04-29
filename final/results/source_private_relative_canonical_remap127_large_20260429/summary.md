# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `all:1536` / `all:1024`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Target | Best no-source | Relative - scalar | Canonical - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `False` | `False` | `False` | `True` | 0.265 | 0.361 | n/a | n/a | 0.442 | 0.250 | 0.250 | n/a | 0.081 |
