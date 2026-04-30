# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:512` / `all:256`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Protected pass | Rotation-sign pass | PQ pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Protected | Rotation-sign | PQ | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Protected - scalar | Rotation-sign - scalar | PQ - scalar | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `True` | `True` | `False` | `False` | `True` | `False` | `False` | `False` | 0.207 | 0.449 | 0.434 | 0.406 | 0.297 | 0.574 | n/a | n/a | n/a | 0.250 | 0.258 | -0.043 | -0.152 | 0.125 | n/a | n/a | n/a |
| 4 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.254 | 0.531 | 0.523 | 0.469 | 0.336 | 0.582 | n/a | n/a | n/a | 0.250 | 0.262 | -0.062 | -0.195 | 0.051 | n/a | n/a | n/a |
| 6 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.336 | 0.570 | 0.547 | 0.504 | 0.340 | 0.598 | n/a | n/a | n/a | 0.250 | 0.277 | -0.066 | -0.230 | 0.027 | n/a | n/a | n/a |
