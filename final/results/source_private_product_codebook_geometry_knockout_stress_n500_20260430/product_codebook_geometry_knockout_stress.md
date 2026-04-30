# Source-Private Product-Codebook Geometry Knockout Stress

- pass gate: `True`
- rows: `18`
- source pass rows: `18`
- adversarial pass rows: `18`
- public-mean pass rows: `3`
- mitigation pass rows: `11`
- mitigation remaps: `[101, 103, 107]`
- max noncanonical source-canonical: `0.022`
- max noncanonical top-mean-removed-canonical: `1.497`
- min noncanonical unique-payload delta vs canonical: `-114`

## Rows

| Remap | Budget | Variant | Source | Target | Best ctrl | Source-ctrl | Source-can | Top worst rem | Top mean rem | Random mean rem | Unique payloads | Collision n | Collision acc | Unique-can | Public pass | Mitigation |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 4 | canonical | 0.482 | 0.250 | 0.268 | 0.214 | 0.000 | 2.069 | 0.103 | 0.017 | 500 | 0 | - | 0 | `False` | `False` |
| 101 | 4 | utility_balanced | 0.494 | 0.250 | 0.258 | 0.236 | 0.012 | 1.803 | 0.180 | 0.033 | 484 | 29 | 0.690 | -16 | `False` | `True` |
| 101 | 4 | opq_procrustes | 0.498 | 0.250 | 0.268 | 0.230 | 0.016 | 1.984 | 0.161 | 0.073 | 498 | 4 | 0.500 | -2 | `False` | `True` |
| 101 | 4 | utility_opq_procrustes | 0.480 | 0.250 | 0.268 | 0.212 | -0.002 | 2.087 | 1.600 | 0.339 | 499 | 2 | 0.500 | -1 | `True` | `True` |
| 101 | 4 | protected_hadamard | 0.498 | 0.250 | 0.268 | 0.230 | 0.016 | 1.387 | 0.105 | 0.032 | 412 | 150 | 0.527 | -88 | `False` | `True` |
| 101 | 4 | utility_protected_hadamard | 0.504 | 0.250 | 0.268 | 0.236 | 0.022 | 1.283 | 0.063 | 0.031 | 386 | 189 | 0.513 | -114 | `False` | `True` |
| 103 | 4 | canonical | 0.508 | 0.250 | 0.264 | 0.244 | 0.000 | 1.961 | 0.202 | 0.000 | 499 | 2 | 0.500 | 0 | `False` | `False` |
| 103 | 4 | utility_balanced | 0.502 | 0.250 | 0.294 | 0.208 | -0.006 | 1.690 | 0.056 | -0.024 | 483 | 30 | 0.567 | -16 | `False` | `False` |
| 103 | 4 | opq_procrustes | 0.498 | 0.250 | 0.274 | 0.224 | -0.010 | 2.000 | 0.121 | -0.024 | 500 | 0 | - | 1 | `False` | `False` |
| 103 | 4 | utility_opq_procrustes | 0.504 | 0.250 | 0.258 | 0.246 | -0.004 | 1.984 | 1.488 | 0.173 | 500 | 0 | - | 1 | `True` | `True` |
| 103 | 4 | protected_hadamard | 0.514 | 0.250 | 0.268 | 0.246 | 0.006 | 1.333 | 0.129 | -0.015 | 404 | 162 | 0.537 | -95 | `False` | `True` |
| 103 | 4 | utility_protected_hadamard | 0.504 | 0.250 | 0.266 | 0.238 | -0.004 | 1.087 | 0.079 | 0.000 | 394 | 177 | 0.497 | -105 | `False` | `True` |
| 107 | 4 | canonical | 0.520 | 0.250 | 0.262 | 0.258 | 0.000 | 1.911 | 0.141 | 0.074 | 498 | 4 | 0.500 | 0 | `False` | `False` |
| 107 | 4 | utility_balanced | 0.516 | 0.250 | 0.246 | 0.270 | -0.004 | 1.692 | 0.113 | 0.015 | 486 | 26 | 0.500 | -12 | `False` | `False` |
| 107 | 4 | opq_procrustes | 0.502 | 0.250 | 0.266 | 0.236 | -0.018 | 1.968 | 0.008 | -0.032 | 499 | 2 | 0.500 | 1 | `False` | `False` |
| 107 | 4 | utility_opq_procrustes | 0.514 | 0.250 | 0.266 | 0.248 | -0.006 | 1.947 | 1.485 | 0.235 | 500 | 0 | - | 2 | `True` | `True` |
| 107 | 4 | protected_hadamard | 0.512 | 0.250 | 0.252 | 0.260 | -0.008 | 1.313 | 0.176 | 0.046 | 425 | 129 | 0.457 | -73 | `False` | `True` |
| 107 | 4 | utility_protected_hadamard | 0.516 | 0.250 | 0.262 | 0.254 | -0.004 | 1.158 | 0.120 | 0.075 | 405 | 164 | 0.494 | -93 | `False` | `True` |

## Interpretation

This gate tests whether OPQ/protected geometry mitigates the lookup-like uniqueness observed in the n500 canonical PQ packet while preserving the source-private control pass and byte-causality diagnostics.

## Pass Rule

A noncanonical geometry variant must pass source controls, keep source accuracy within 0.02 of canonical PQ at the same remap/budget, and either improve public-mean top-codeword lift removal by >=0.05 or reduce unique matched payloads by at least 25 at n500 while the reused-payload subset still beats target by >=0.10.
