# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Protected pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Protected | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Protected - scalar | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `True` | `False` | `True` | `False` | `False` | `False` | 0.207 | 0.418 | 0.393 | 0.432 | n/a | 0.350 | n/a | 0.250 | 0.250 | 0.014 | n/a | -0.068 | n/a |
| 4 | `False` | `True` | `True` | `True` | `False` | `False` | `False` | 0.275 | 0.445 | 0.453 | 0.436 | n/a | 0.506 | n/a | 0.250 | 0.250 | -0.010 | n/a | 0.061 | n/a |
| 6 | `False` | `True` | `True` | `True` | `False` | `False` | `False` | 0.350 | 0.492 | 0.457 | 0.453 | n/a | 0.506 | n/a | 0.250 | 0.266 | -0.039 | n/a | 0.014 | n/a |
