# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `True` | 0.953 | 0.979 | 0.250 | 0.250 | -0.025 |
