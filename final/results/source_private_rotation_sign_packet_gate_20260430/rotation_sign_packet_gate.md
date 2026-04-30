# Source-Private Rotation-Sign Packet Gate

- pass gate: `False`
- rows: `9`
- pass rows: `0`
- remaps with pass: `[]`
- remap seeds: `[101, 103, 107]`
- budgets: `[2, 4, 6]`
- max rotation-sign accuracy: `0.348`
- min passing rotation-control margin: `-`
- max passing p50 decode latency ms: `-`

## Rows

| Remap | Budget | N | Rotation-sign | Raw sign | Scalar WZ | Protected | QJL | Target | Best rotation control | Rotation-control | Rotation-scalar | p50 ms | Rotation pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 256 | 0.297 | 0.297 | 0.449 | 0.406 | 0.434 | 0.250 | 0.297 | 0.000 | -0.152 | 0.580 | `False` |
| 101 | 4 | 256 | 0.336 | 0.336 | 0.531 | 0.469 | 0.523 | 0.250 | 0.277 | 0.059 | -0.195 | 6.681 | `False` |
| 101 | 6 | 256 | 0.340 | 0.340 | 0.570 | 0.504 | 0.547 | 0.250 | 0.352 | -0.012 | -0.230 | 6.922 | `False` |
| 103 | 2 | 256 | 0.312 | 0.312 | 0.469 | 0.461 | 0.430 | 0.250 | 0.289 | 0.023 | -0.156 | 0.573 | `False` |
| 103 | 4 | 256 | 0.348 | 0.348 | 0.500 | 0.473 | 0.535 | 0.250 | 0.281 | 0.066 | -0.152 | 6.636 | `False` |
| 103 | 6 | 256 | 0.332 | 0.332 | 0.539 | 0.535 | 0.531 | 0.250 | 0.332 | 0.000 | -0.207 | 9.520 | `False` |
| 107 | 2 | 256 | 0.309 | 0.309 | 0.449 | 0.484 | 0.449 | 0.250 | 0.297 | 0.012 | -0.141 | 0.581 | `False` |
| 107 | 4 | 256 | 0.316 | 0.316 | 0.488 | 0.496 | 0.500 | 0.250 | 0.281 | 0.035 | -0.172 | 7.964 | `False` |
| 107 | 6 | 256 | 0.332 | 0.332 | 0.520 | 0.492 | 0.504 | 0.250 | 0.340 | -0.008 | -0.188 | 9.399 | `False` |

## Interpretation

This gate isolates a compression-native packet: the source sends only signs of random projections of its private evidence vector, and the target decodes by Hamming distance against public candidate side information. It is a publishable systems/codec contribution only if the same-bit constrained-shuffle, answer-masked, permuted-bit, and random controls collapse to target accuracy while the matched packet remains useful.
