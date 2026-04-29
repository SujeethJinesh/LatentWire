# Source-Private Relative Score Bootstrap

- pass gate: `True`
- method condition: `relative_canonical_score_source`
- budget bytes: `4`
- bootstrap samples: `2000`
- mean relative accuracy: `0.442`
- mean relative minus scalar: `0.081`
- mean remap relative minus scalar: `0.081`
- min relative vs target CI95 low: `0.152`
- min remap relative vs scalar CI95 low: `0.053`

| Result | Remap | Relative | Scalar | Target | Raw sign | Relative - scalar CI95 | Relative - target CI95 | Bytes rel/scalar | p50 rel/scalar ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_private_relative_canonical_remap127_large_20260429` | `127` | 0.442 | 0.361 | 0.250 | 0.327 | [0.053, 0.110] | [0.152, 0.233] | 4.0/4.0 | 4.33/4.18 |

Pass rule: Relative score packet should keep positive paired lower bound versus target-only and positive mean remap delta versus scalar at equal actual bytes.
