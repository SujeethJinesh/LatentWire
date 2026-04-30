# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:512` / `all:256`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Protected pass | Rotation-sign pass | PQ pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Protected | Rotation-sign | PQ | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Protected - scalar | Rotation-sign - scalar | PQ - scalar | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | `True` | `True` | `True` | `False` | `False` | `False` | `False` | `False` | 0.262 | 0.449 | 0.449 | 0.484 | 0.309 | 0.512 | n/a | n/a | n/a | 0.250 | 0.258 | 0.035 | -0.141 | 0.062 | n/a | n/a | n/a |
| 4 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.270 | 0.488 | 0.500 | 0.496 | 0.316 | 0.527 | n/a | n/a | n/a | 0.250 | 0.258 | 0.008 | -0.172 | 0.039 | n/a | n/a | n/a |
| 6 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.352 | 0.520 | 0.504 | 0.492 | 0.332 | 0.523 | n/a | n/a | n/a | 0.250 | 0.262 | -0.027 | -0.188 | 0.004 | n/a | n/a | n/a |
