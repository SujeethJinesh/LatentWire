# Conditional PQ Basis/Schema Grid

- pass gate: `False`
- rows: `28`
- pass rows: `0`
- bidirectional basis/mode passes: `0`
- max source accuracy: `0.316406`
- max source minus best control: `0.007812000000000041`
- max CI95 low vs best control: `0.0`

## Best Rows

| Mode | Basis | Direction | Pass | Source | Target | Best control | CI95 low | Unquantized |
|---|---|---|---:|---:|---:|---:|---:|---:|
| legacy | diag_only | holdout_to_core | `False` | 0.316 | 0.250 | 0.309 | -0.031 | 0.512 |
| legacy | semantic | holdout_to_core | `False` | 0.254 | 0.250 | 0.250 | 0.000 | 0.445 |
| plausible_decoys | slot | core_to_holdout | `False` | 0.309 | 0.250 | 0.309 | 0.000 | 0.277 |
| legacy | anchor_relative | holdout_to_core | `False` | 0.270 | 0.250 | 0.270 | -0.020 | 0.445 |
| plausible_decoys | anchor_relative | holdout_to_core | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.605 |
| plausible_decoys | semantic | core_to_holdout | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.371 |
| plausible_decoys | semantic | holdout_to_core | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.492 |
| plausible_decoys | diag_only | core_to_holdout | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.629 |

## Interpretation

No existing public basis or diagnostic-table mode rescues bidirectional held-out-family conditional PQ innovation at n256. This weakens the hypothesis that cross-family failure is only a static-basis selection problem and points to ontology calibration or public-conditioned codebooks as the next branch.
