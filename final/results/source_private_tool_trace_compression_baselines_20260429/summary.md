# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `all:512` / `all:256`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `False` | 0.934 | 0.973 | 0.250 | 0.250 | -0.039 |
| 12 | `False` | `False` | 0.992 | 1.000 | 0.250 | 0.250 | -0.008 |
