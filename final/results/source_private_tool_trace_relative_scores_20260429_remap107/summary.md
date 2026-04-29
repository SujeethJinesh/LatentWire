# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Syndrome | Scalar | QJL | Relative | Target | Best no-source | Relative - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `True` | `False` | `True` | 0.350 | 0.492 | n/a | 0.506 | 0.250 | 0.250 | 0.014 |
