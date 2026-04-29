# Source-Private Tool-Trace Compression Baselines

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `True` | `False` | 0.414 | 0.125 | 0.250 | 0.258 | 0.129 |
