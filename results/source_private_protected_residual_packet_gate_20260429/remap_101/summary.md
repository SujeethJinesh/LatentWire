# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Protected pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Protected | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Protected - scalar | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `True` | `False` | `True` | `False` | `False` | `False` | 0.193 | 0.418 | 0.396 | 0.430 | n/a | 0.350 | n/a | 0.250 | 0.250 | 0.012 | n/a | -0.068 | n/a |
| 4 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | 0.271 | 0.432 | 0.461 | 0.447 | n/a | 0.494 | n/a | 0.250 | 0.250 | 0.016 | n/a | 0.062 | n/a |
| 6 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | 0.355 | 0.463 | 0.447 | 0.447 | n/a | 0.494 | n/a | 0.250 | 0.268 | -0.016 | n/a | 0.031 | n/a |
