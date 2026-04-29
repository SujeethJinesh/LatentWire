# Source-Private Relative Score Bootstrap

- pass gate: `False`
- method condition: `relative_canonical_score_source`
- budget bytes: `4`
- bootstrap samples: `2000`
- mean relative accuracy: `0.490`
- mean relative minus scalar: `0.037`
- mean remap relative minus scalar: `0.037`
- min relative vs target CI95 low: `0.146`
- min remap relative vs scalar CI95 low: `-0.035`

| Result | Remap | Relative | Scalar | Target | Raw sign | Relative - scalar CI95 | Relative - target CI95 | Bytes rel/scalar | p50 rel/scalar ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_private_relative_canonical_remap101_20260429` | `101` | 0.494 | 0.426 | 0.250 | 0.326 | [0.029, 0.107] | [0.184, 0.303] | 4.0/4.0 | 3.87/3.76 |
| `source_private_relative_canonical_remap103_20260429` | `103` | 0.520 | 0.496 | 0.250 | 0.328 | [-0.014, 0.061] | [0.213, 0.328] | 4.0/4.0 | 3.61/3.94 |
| `source_private_relative_canonical_remap107_20260429` | `107` | 0.506 | 0.502 | 0.250 | 0.326 | [-0.035, 0.043] | [0.199, 0.311] | 4.0/4.0 | 3.65/3.81 |
| `source_private_relative_canonical_remap109_20260429` | `109` | 0.477 | 0.451 | 0.250 | 0.342 | [-0.008, 0.061] | [0.170, 0.281] | 4.0/4.0 | 3.65/3.95 |
| `source_private_relative_canonical_remap113_20260429` | `113` | 0.473 | 0.436 | 0.250 | 0.330 | [0.002, 0.072] | [0.164, 0.279] | 4.0/4.0 | 4.22/4.39 |
| `source_private_relative_canonical_remap127_20260429` | `127` | 0.453 | 0.428 | 0.250 | 0.314 | [-0.010, 0.061] | [0.146, 0.262] | 4.0/4.0 | 4.15/4.00 |
| `source_private_relative_canonical_remap131_20260429` | `131` | 0.506 | 0.434 | 0.250 | 0.328 | [0.035, 0.109] | [0.197, 0.311] | 4.0/4.0 | 4.02/3.78 |

Pass rule: Relative score packet should keep positive paired lower bound versus target-only and positive mean remap delta versus scalar at equal actual bytes.
