# Source-Private Product-Codebook Geometry Gate

- pass gate: `False`
- rows: `2`
- source pass rows: `1`
- promoted rows: `0`
- max source accuracy: `0.527`
- max noncanonical minus canonical: `0.016`

| Remap | Budget | Variant | Source | Target | Best control | Source-control | Source-canonical | Pass |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 107 | 2 | canonical | 0.512 | 0.250 | 0.312 | 0.199 | 0.000 | `False` |
| 107 | 2 | opq_procrustes | 0.527 | 0.250 | 0.289 | 0.238 | 0.016 | `True` |

Pass rule: A non-canonical geometry variant must pass source controls and beat canonical contiguous PQ by at least +0.03 accuracy at the same remap, budget, and exact eval IDs.
