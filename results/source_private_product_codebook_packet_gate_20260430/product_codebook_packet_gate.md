# Source-Private Product-Codebook Packet Gate

- pass gate: `False`
- rows: `9`
- pass rows: `0`
- remaps with pass: `[]`
- functional pass gate: `True`
- functional pass rows: `8`
- functional remaps with pass: `[101, 103, 107]`
- systems latency pass gate: `False`
- systems pass rows: `0`
- remap seeds: `[101, 103, 107]`
- budgets: `[2, 4, 6]`
- max product-codebook accuracy: `0.598`
- min passing product-codebook-control margin: `-`
- min passing product-codebook-scalar margin: `-`
- max passing p50 decode latency ms: `-`

## Rows

| Remap | Budget | N | Product codebook | Scalar WZ | Protected | QJL | Rotation-sign | Target | Best PQ control | PQ-control | PQ-scalar | p50 ms | PQ pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 256 | 0.574 | 0.449 | 0.406 | 0.434 | 0.297 | 0.250 | 0.254 | 0.320 | 0.125 | 10.285 | `True` |
| 101 | 4 | 256 | 0.582 | 0.531 | 0.469 | 0.523 | 0.336 | 0.250 | 0.281 | 0.301 | 0.051 | 9.659 | `True` |
| 101 | 6 | 256 | 0.598 | 0.570 | 0.504 | 0.547 | 0.340 | 0.250 | 0.293 | 0.305 | 0.027 | 9.944 | `True` |
| 103 | 2 | 256 | 0.539 | 0.469 | 0.461 | 0.430 | 0.312 | 0.250 | 0.285 | 0.254 | 0.070 | 9.039 | `True` |
| 103 | 4 | 256 | 0.531 | 0.500 | 0.473 | 0.535 | 0.348 | 0.250 | 0.293 | 0.238 | 0.031 | 9.415 | `True` |
| 103 | 6 | 256 | 0.551 | 0.539 | 0.535 | 0.531 | 0.332 | 0.250 | 0.273 | 0.277 | 0.012 | 9.499 | `True` |
| 107 | 2 | 256 | 0.512 | 0.449 | 0.484 | 0.449 | 0.309 | 0.250 | 0.312 | 0.199 | 0.062 | 8.313 | `False` |
| 107 | 4 | 256 | 0.527 | 0.488 | 0.496 | 0.500 | 0.316 | 0.250 | 0.293 | 0.234 | 0.039 | 9.005 | `True` |
| 107 | 6 | 256 | 0.523 | 0.520 | 0.492 | 0.504 | 0.332 | 0.250 | 0.281 | 0.242 | 0.004 | 9.759 | `True` |

## Interpretation

This gate tests whether a product-quantized discrete packet can become a compression-native replacement for scalar Wyner-Ziv: each byte is a learned centroid index for one subspace of the source-projected vector. It is a method contribution only if the code indices preserve source-private candidate margins without being explained by label-shuffled, constrained-shuffled, answer-masked, permuted-code, or random controls. The aggregate reports functional pass separately from systems-latency pass because the current implementation uses a simple Python decoder rather than an optimized table-lookup kernel.
