# Source-Private Tool-Trace Compression Baselines

- pass gate: `True`
- train/eval: `all:768` / `all:500`
- candidate view: `slot`
- exact ID parity: `True`

| Budget bytes | Learned > compression | Scalar pass | QJL pass | Protected pass | Rotation-sign pass | PQ pass | Relative pass | Canonical relative pass | Consistent posterior pass | Syndrome | Scalar | QJL | Protected | Rotation-sign | PQ | Relative | Canonical relative | Consistent posterior | Target | Best no-source | Protected - scalar | Rotation-sign - scalar | PQ - scalar | Relative - scalar | Canonical - scalar | Consistent - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | `True` | `True` | `True` | `False` | `True` | `False` | `False` | `False` | 0.252 | 0.502 | 0.480 | 0.474 | 0.326 | 0.508 | n/a | n/a | n/a | 0.250 | 0.252 | -0.028 | -0.176 | 0.006 | n/a | n/a | n/a |
