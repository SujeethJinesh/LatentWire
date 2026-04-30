# Source-Private Product-Codebook Paired Uncertainty

- pass gate: `True`
- rows: `3`
- pass rows: `3`
- remaps with pass: `[101, 103, 107]`
- min passing CI95 low vs target: `0.174`
- min passing CI95 low vs best control: `0.154`
- min CI95 low vs scalar: `-0.032`
- max product-codebook accuracy: `0.520`

## Rows

| Remap | Budget | N | PQ | Target | Best control | Scalar | PQ-target | PQ-control CI low | PQ-target CI low | PQ-scalar CI low | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 4 | 500 | 0.482 | 0.250 | 0.268 | 0.424 | 0.232 | 0.154 | 0.174 | 0.022 | `True` |
| 103 | 4 | 500 | 0.508 | 0.250 | 0.262 | 0.502 | 0.258 | 0.186 | 0.198 | -0.032 | `True` |
| 107 | 4 | 500 | 0.520 | 0.250 | 0.252 | 0.504 | 0.270 | 0.210 | 0.212 | -0.024 | `True` |

## Interpretation

This summary tests whether product-codebook gains are stable at the example-paired level rather than only aggregate accuracies. It is intentionally stricter against source-destroying controls than against scalar WZ: PQ must prove source-causal lift, while scalar WZ remains a strong adjacent codec comparator.

## Pass Rule

At least one row per remapped codebook must have exact ID parity, paired CI95 low >0.15 versus target-only, paired CI95 low >0.10 versus the best product-codebook destructive control, and stay within 0.02 accuracy of scalar Wyner-Ziv.
