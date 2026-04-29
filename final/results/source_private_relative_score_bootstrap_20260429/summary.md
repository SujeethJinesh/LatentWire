# Source-Private Relative Score Bootstrap

- pass gate: `True`
- budget bytes: `4`
- bootstrap samples: `2000`
- mean relative accuracy: `0.630`
- mean relative minus scalar: `0.024`
- mean remap relative minus scalar: `0.032`
- min relative vs target CI95 low: `0.189`
- min remap relative vs scalar CI95 low: `-0.035`

| Result | Remap | Relative | Scalar | Target | Raw sign | Relative - scalar CI95 | Relative - target CI95 | Bytes rel/scalar | p50 rel/scalar ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_private_tool_trace_relative_scores_20260429_seed29_30_budget4` | `None` | 1.000 | 1.000 | 0.250 | 0.316 | [0.000, 0.000] | [0.713, 0.787] | 4.0/4.0 | 25.27/25.06 |
| `source_private_tool_trace_relative_scores_20260429_remap101_budget4` | `101` | 0.494 | 0.426 | 0.250 | 0.326 | [0.033, 0.105] | [0.189, 0.303] | 4.0/4.0 | 22.70/24.23 |
| `source_private_tool_trace_relative_scores_20260429_remap103_budget4` | `103` | 0.520 | 0.496 | 0.250 | 0.328 | [-0.012, 0.061] | [0.209, 0.326] | 4.0/4.0 | 22.78/23.73 |
| `source_private_tool_trace_relative_scores_20260429_remap107_budget4` | `107` | 0.506 | 0.502 | 0.250 | 0.326 | [-0.035, 0.043] | [0.197, 0.311] | 4.0/4.0 | 24.03/24.27 |

Pass rule: Relative score packet should keep positive paired lower bound versus target-only and positive mean remap delta versus scalar at equal actual bytes.
