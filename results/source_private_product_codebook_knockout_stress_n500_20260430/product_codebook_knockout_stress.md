# Source-Private Product-Codebook Knockout Stress

- pass gate: `True`
- adversarial pass gate: `True`
- public-mean pass gate: `False`
- rows: `3`
- adversarial pass rows: `3`
- public-mean pass rows: `0`
- adversarial remaps with pass: `[101, 103, 107]`
- public-mean remaps with pass: `[]`
- min top-worst lift removed fraction: `1.911`
- min top-mean lift removed fraction: `0.103`
- max random-mean lift removed fraction: `0.074`
- min source-target lift: `0.232`
- min unique payloads: `498`
- max payload frequency: `2`
- min payload entropy bits: `8.958`

## Rows

| Remap | Budget | N | Source | Target | Top worst | Top mean | Random mean | Random random | Top only | Mean payload | Top worst lift removed | Top mean lift removed | Random mean lift removed | Adv pass | Public pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 4 | 500 | 0.482 | 0.250 | 0.002 | 0.458 | 0.478 | 0.456 | 0.566 | 0.220 | 2.069 | 0.103 | 0.017 | `True` | `False` |
| 103 | 4 | 500 | 0.508 | 0.250 | 0.002 | 0.456 | 0.508 | 0.504 | 0.648 | 0.264 | 1.961 | 0.202 | 0.000 | `True` | `False` |
| 107 | 4 | 500 | 0.520 | 0.250 | 0.004 | 0.482 | 0.500 | 0.454 | 0.586 | 0.234 | 1.911 | 0.141 | 0.074 | `True` | `False` |

## Interpretation

This is a diagnostic stress test for the n500 product-codebook packet. The source packet is one centroid index per byte. For each example, the analyzer decomposes the target-side PQ distance margin by byte, selects the byte that most helps the gold candidate against the nearest wrong candidate, and then corrupts that byte. The adversarial replacement is an oracle analysis, not a deployable control; the public-mean replacement is a stronger source-erasure test because it uses only train/public codebook statistics.

## Pass Rule

Adversarial pass: every remapped codebook must have a matched-source lift >=0.15 over target-only, exact ID parity, and top-margin oracle codeword replacement must remove at least 50% of matched lift with paired CI95 low >0.05 versus the matched packet. Public-mean pass is stricter: replacing the same top-margin byte with a train-public mean code must remove at least 25% of lift and at least 5 points more lift fraction than a random-subspace public-mean replacement.
