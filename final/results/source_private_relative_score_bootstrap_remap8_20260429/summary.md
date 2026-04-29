# Source-Private Relative Score Bootstrap

- pass gate: `False`
- budget bytes: `4`
- bootstrap samples: `5000`
- mean relative accuracy: `0.553`
- mean relative minus scalar: `0.032`
- mean remap relative minus scalar: `0.037`
- min relative vs target CI95 low: `0.146`
- min remap relative vs scalar CI95 low: `-0.035`

| Result | Remap | Relative | Scalar | Target | Raw sign | Relative - scalar CI95 | Relative - target CI95 | Bytes rel/scalar | p50 rel/scalar ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_private_tool_trace_relative_scores_20260429_seed29_30_budget4` | `None` | 1.000 | 1.000 | 0.250 | 0.316 | [0.000, 0.000] | [0.713, 0.787] | 4.0/4.0 | 25.27/25.06 |
| `source_private_tool_trace_relative_scores_20260429_remap101_budget4` | `101` | 0.494 | 0.426 | 0.250 | 0.326 | [0.033, 0.105] | [0.188, 0.299] | 4.0/4.0 | 22.70/24.23 |
| `source_private_tool_trace_relative_scores_20260429_remap103_budget4` | `103` | 0.520 | 0.496 | 0.250 | 0.328 | [-0.014, 0.061] | [0.209, 0.328] | 4.0/4.0 | 22.78/23.73 |
| `source_private_tool_trace_relative_scores_20260429_remap107_budget4` | `107` | 0.506 | 0.502 | 0.250 | 0.326 | [-0.035, 0.043] | [0.195, 0.314] | 4.0/4.0 | 24.03/24.27 |
| `source_private_tool_trace_relative_scores_20260429_remap109_budget4` | `109` | 0.477 | 0.451 | 0.250 | 0.342 | [-0.010, 0.061] | [0.170, 0.283] | 4.0/4.0 | 47.23/46.86 |
| `source_private_tool_trace_relative_scores_20260429_remap113_budget4` | `113` | 0.473 | 0.436 | 0.250 | 0.330 | [0.000, 0.072] | [0.164, 0.281] | 4.0/4.0 | 47.35/44.44 |
| `source_private_tool_trace_relative_scores_20260429_remap127_budget4` | `127` | 0.453 | 0.428 | 0.250 | 0.314 | [-0.010, 0.061] | [0.146, 0.264] | 4.0/4.0 | 9.91/9.61 |
| `source_private_tool_trace_relative_scores_20260429_remap131_budget4` | `131` | 0.506 | 0.434 | 0.250 | 0.328 | [0.035, 0.109] | [0.199, 0.312] | 4.0/4.0 | 10.02/9.61 |

Pass rule: Relative score packet should keep positive paired lower bound versus target-only and positive mean remap delta versus scalar at equal actual bytes.
