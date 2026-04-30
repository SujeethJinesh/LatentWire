# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:512` / `all:256`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Protected pass | Rotation-sign pass | PQ pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Protected | Rotation-sign | PQ | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Protected - scalar | Rotation-sign - scalar | PQ - scalar | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.195 | 0.469 | 0.430 | 0.461 | 0.312 | 0.539 | n/a | n/a | n/a | 0.250 | 0.270 | -0.008 | -0.156 | 0.070 | n/a | n/a | n/a |
| 4 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.281 | 0.500 | 0.535 | 0.473 | 0.348 | 0.531 | n/a | n/a | n/a | 0.250 | 0.273 | -0.027 | -0.152 | 0.031 | n/a | n/a | n/a |
| 6 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.332 | 0.539 | 0.531 | 0.535 | 0.332 | 0.551 | n/a | n/a | n/a | 0.250 | 0.262 | -0.004 | -0.207 | 0.012 | n/a | n/a | n/a |
