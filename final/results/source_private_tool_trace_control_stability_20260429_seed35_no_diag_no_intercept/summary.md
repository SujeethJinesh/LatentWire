# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `all:768` / `all:512`
- candidate view: `no_diag`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `False` | 0.252 | 1.000 | 0.250 | 0.262 | -0.748 |
