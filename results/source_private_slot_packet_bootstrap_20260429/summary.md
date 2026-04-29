# Source-Private Slot Packet Bootstrap

- pass gate: `True`
- budget bytes: `6`
- bootstrap samples: `2000`
- mean scalar accuracy: `0.808`
- mean scalar minus best strict control: `0.552`
- min target delta CI95 low: `0.156`
- min raw-sign delta CI95 low: `0.072`

| Result | Pass | Remap | Scalar | Target | Best strict control | Raw sign | Delta target CI95 | Delta raw CI95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_private_tool_trace_control_stability_20260429_seed29_slot_no_intercept` | `True` | `None` | 1.000 | 0.250 | 0.250 | 0.307 | [0.713, 0.787] | [0.654, 0.732] |
| `source_private_tool_trace_control_stability_20260429_seed31_slot_no_intercept` | `True` | `None` | 1.000 | 0.250 | 0.258 | 0.188 | [0.715, 0.787] | [0.777, 0.846] |
| `source_private_tool_trace_control_stability_20260429_seed33_slot_no_intercept` | `True` | `None` | 1.000 | 0.250 | 0.250 | 0.207 | [0.713, 0.787] | [0.756, 0.828] |
| `source_private_tool_trace_control_stability_20260429_seed35_slot_no_intercept` | `True` | `None` | 1.000 | 0.250 | 0.262 | 0.182 | [0.713, 0.787] | [0.783, 0.852] |
| `source_private_tool_trace_control_stability_20260429_seed37_slot_no_intercept` | `True` | `None` | 1.000 | 0.250 | 0.250 | 0.201 | [0.711, 0.787] | [0.764, 0.834] |
| `source_private_tool_trace_slot_remap_20260429_seed101` | `True` | `101` | 0.463 | 0.250 | 0.264 | 0.332 | [0.156, 0.270] | [0.072, 0.189] |
| `source_private_tool_trace_slot_remap_20260429_seed103` | `True` | `103` | 0.508 | 0.250 | 0.266 | 0.316 | [0.199, 0.314] | [0.127, 0.252] |
| `source_private_tool_trace_slot_remap_20260429_seed107` | `True` | `107` | 0.492 | 0.250 | 0.250 | 0.330 | [0.186, 0.303] | [0.104, 0.221] |

Pass rule: Every row should have paired bootstrap lower bound >0.15 versus target-only, and mean scalar-minus-best-strict-control should exceed 0.15. Raw sign sketch is reported as a compression baseline but not part of strict no-source controls.
