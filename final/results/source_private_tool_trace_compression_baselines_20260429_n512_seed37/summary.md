# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `all:768` / `all:512`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `False` | 0.945 | 0.986 | 0.250 | 0.250 | -0.041 |
