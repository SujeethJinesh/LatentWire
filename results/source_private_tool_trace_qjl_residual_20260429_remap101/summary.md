# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Syndrome | Scalar | QJL | Target | Best no-source | QJL - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `True` | `True` | 0.355 | 0.463 | 0.447 | 0.250 | 0.264 | -0.016 |
