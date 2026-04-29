# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Target | Best no-source | Relative - scalar | Canonical - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `True` | `False` | `False` | `True` | 0.281 | 0.451 | n/a | n/a | 0.477 | 0.250 | 0.250 | n/a | 0.025 |
