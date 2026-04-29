# Anchor-Relative Sparse Packet Direction

- direction: `core_to_holdout`
- pass gate: `False`
- train/eval families: `core -> holdout`
- candidate view: `semantic`

| Budget | Sparse | Target | Best control | Sparse-target | Sparse-control | Controls ok | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.242 | 0.250 | 0.453 | -0.008 | -0.211 | `False` | `False` |
| 4 | 0.250 | 0.250 | 0.271 | 0.000 | -0.021 | `True` | `False` |
| 6 | 0.125 | 0.250 | 0.283 | -0.125 | -0.158 | `False` | `False` |
| 8 | 0.250 | 0.250 | 0.375 | 0.000 | -0.125 | `False` | `False` |
