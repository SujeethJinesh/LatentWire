# Source-Private Product-Codebook Geometry Gate

- pass gate: `False`
- rows: `3`
- source pass rows: `2`
- promoted rows: `0`
- max source accuracy: `0.582`
- max noncanonical minus canonical: `-0.004`

| Remap | Budget | Variant | Source | Target | Best control | Source-control | Source-canonical | Pass |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | 4 | canonical | 0.582 | 0.250 | 0.254 | 0.328 | 0.000 | `True` |
| 101 | 4 | opq_procrustes | 0.578 | 0.250 | 0.270 | 0.309 | -0.004 | `True` |
| 101 | 4 | utility_opq_procrustes | 0.578 | 0.250 | 0.316 | 0.262 | -0.004 | `False` |

Pass rule: A non-canonical geometry variant must pass source controls and beat canonical contiguous PQ by at least +0.03 accuracy at the same remap, budget, and exact eval IDs.
