# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:500`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Protected pass | Rotation-sign pass | PQ pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Protected | Rotation-sign | PQ | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Protected - scalar | Rotation-sign - scalar | PQ - scalar | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.272 | 0.424 | 0.424 | 0.454 | 0.324 | 0.482 | n/a | n/a | n/a | 0.250 | 0.254 | 0.030 | -0.100 | 0.058 | n/a | n/a | n/a |
