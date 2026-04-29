# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:512` / `all:256`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `True` | 0.910 | 0.945 | 0.250 | 0.250 | -0.035 |
