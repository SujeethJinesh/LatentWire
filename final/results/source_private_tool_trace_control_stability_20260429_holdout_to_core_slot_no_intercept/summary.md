# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `holdout:768` / `core:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | `False` | `True` | 0.033 | 0.625 | 0.250 | 0.250 | -0.592 |
