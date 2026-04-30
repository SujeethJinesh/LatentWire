# Source-Private Product-Codebook Paired Uncertainty

- pass gate: `True`
- rows: `9`
- pass rows: `8`
- remaps with pass: `[101, 103, 107]`
- min passing CI95 low vs target: `0.191`
- min passing CI95 low vs best control: `0.152`
- min CI95 low vs scalar: `-0.035`
- max product-codebook accuracy: `0.598`

## Rows

| Remap | Budget | N | PQ | Target | Best control | Scalar | PQ-target | PQ-control CI low | PQ-target CI low | PQ-scalar CI low | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 256 | 0.574 | 0.250 | 0.254 | 0.449 | 0.324 | 0.234 | 0.238 | 0.066 | `True` |
| 101 | 4 | 256 | 0.582 | 0.250 | 0.281 | 0.531 | 0.332 | 0.219 | 0.250 | -0.004 | `True` |
| 101 | 6 | 256 | 0.598 | 0.250 | 0.293 | 0.570 | 0.348 | 0.230 | 0.266 | -0.016 | `True` |
| 103 | 2 | 256 | 0.539 | 0.250 | 0.285 | 0.469 | 0.289 | 0.172 | 0.203 | 0.004 | `True` |
| 103 | 4 | 256 | 0.531 | 0.250 | 0.293 | 0.500 | 0.281 | 0.152 | 0.195 | -0.023 | `True` |
| 103 | 6 | 256 | 0.551 | 0.250 | 0.273 | 0.539 | 0.301 | 0.191 | 0.211 | -0.035 | `True` |
| 107 | 2 | 256 | 0.512 | 0.250 | 0.312 | 0.449 | 0.262 | 0.117 | 0.180 | 0.004 | `False` |
| 107 | 4 | 256 | 0.527 | 0.250 | 0.293 | 0.488 | 0.277 | 0.152 | 0.195 | -0.016 | `True` |
| 107 | 6 | 256 | 0.523 | 0.250 | 0.281 | 0.520 | 0.273 | 0.160 | 0.191 | -0.035 | `True` |

## Interpretation

This summary tests whether product-codebook gains are stable at the example-paired level rather than only aggregate accuracies. It is intentionally stricter against source-destroying controls than against scalar WZ: PQ must prove source-causal lift, while scalar WZ remains a strong adjacent codec comparator.

## Pass Rule

At least one row per remapped codebook must have exact ID parity, paired CI95 low >0.15 versus target-only, paired CI95 low >0.10 versus the best product-codebook destructive control, and stay within 0.02 accuracy of scalar Wyner-Ziv.
