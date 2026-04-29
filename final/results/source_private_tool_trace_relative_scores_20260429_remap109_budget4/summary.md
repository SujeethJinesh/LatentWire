# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Syndrome | Scalar | QJL | Relative | Target | Best no-source | Relative - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `True` | `False` | `True` | 0.281 | 0.451 | n/a | 0.477 | 0.250 | 0.250 | 0.025 |
