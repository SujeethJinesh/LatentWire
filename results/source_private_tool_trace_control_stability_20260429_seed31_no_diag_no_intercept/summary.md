# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `no_diag`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `True` | 0.238 | 0.998 | 0.250 | 0.250 | -0.760 |
