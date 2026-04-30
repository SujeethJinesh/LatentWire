# Source-Private Product-Codebook Geometry Gate

- pass gate: `False`
- rows: `4`
- source pass rows: `4`
- promoted rows: `0`
- max source accuracy: `0.586`
- max noncanonical minus canonical: `0.004`

| Remap | Budget | Variant | Source | Target | Best control | Source-control | Source-canonical | Pass |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | 4 | canonical | 0.582 | 0.250 | 0.254 | 0.328 | 0.000 | `True` |
| 101 | 4 | utility_round_robin | 0.574 | 0.250 | 0.297 | 0.277 | -0.008 | `True` |
| 101 | 4 | utility_balanced | 0.582 | 0.250 | 0.289 | 0.293 | 0.000 | `True` |
| 101 | 4 | random_balanced | 0.586 | 0.250 | 0.289 | 0.297 | 0.004 | `True` |

Pass rule: A non-canonical geometry variant must pass source controls and beat canonical contiguous PQ by at least +0.03 accuracy at the same remap, budget, and exact eval IDs.
