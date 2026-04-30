# Source-Private Product-Codebook Packet Gate

- pass gate: `True`
- rows: `3`
- pass rows: `3`
- remaps with pass: `[101, 103, 107]`
- functional pass gate: `True`
- functional pass rows: `3`
- functional remaps with pass: `[101, 103, 107]`
- systems latency pass gate: `True`
- systems pass rows: `3`
- remap seeds: `[101, 103, 107]`
- budgets: `[4]`
- max product-codebook accuracy: `0.520`
- min passing product-codebook-control margin: `0.214`
- min passing product-codebook-scalar margin: `0.006`
- max passing p50 decode latency ms: `0.378`

## Rows

| Remap | Budget | N | Product codebook | Scalar WZ | Protected | QJL | Rotation-sign | Target | Best PQ control | PQ-control | PQ-scalar | p50 ms | PQ pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 4 | 500 | 0.482 | 0.424 | 0.454 | 0.424 | 0.324 | 0.250 | 0.268 | 0.214 | 0.058 | 0.360 | `True` |
| 103 | 4 | 500 | 0.508 | 0.502 | 0.474 | 0.480 | 0.326 | 0.250 | 0.262 | 0.246 | 0.006 | 0.378 | `True` |
| 107 | 4 | 500 | 0.520 | 0.504 | 0.442 | 0.476 | 0.324 | 0.250 | 0.252 | 0.268 | 0.016 | 0.368 | `True` |

## Interpretation

This gate tests whether a product-quantized discrete packet can become a compression-native replacement for scalar Wyner-Ziv: each byte is a learned centroid index for one subspace of the source-projected vector. It is a method contribution only if the code indices preserve source-private candidate margins without being explained by label-shuffled, constrained-shuffled, answer-masked, permuted-code, or random controls. The aggregate reports functional pass separately from systems-latency pass because the current implementation uses a simple Python decoder rather than an optimized table-lookup kernel.
