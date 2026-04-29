# Anchor-Relative Sparse Packet Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- candidate view: `semantic`

| Budget | Sparse | Target | Best control | Sparse-target | Sparse-control | Controls ok | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.496 | 0.250 | 0.250 | 0.246 | 0.246 | `True` | `True` |
| 4 | 0.248 | 0.250 | 0.250 | -0.002 | -0.002 | `True` | `False` |
| 6 | 0.270 | 0.250 | 0.250 | 0.020 | 0.020 | `True` | `False` |
| 8 | 0.373 | 0.250 | 0.262 | 0.123 | 0.111 | `True` | `True` |
