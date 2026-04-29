# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `True` | `True` | `False` | `False` | `False` | 0.172 | 0.436 | 0.439 | n/a | 0.363 | n/a | 0.250 | 0.250 | n/a | -0.072 | n/a |
| 4 | `False` | `True` | `True` | `False` | `True` | `False` | 0.254 | 0.475 | 0.461 | n/a | 0.520 | n/a | 0.250 | 0.250 | n/a | 0.045 | n/a |
| 6 | `False` | `True` | `True` | `False` | `True` | `False` | 0.361 | 0.508 | 0.484 | n/a | 0.520 | n/a | 0.250 | 0.250 | n/a | 0.012 | n/a |
