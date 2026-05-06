# Hybrid Architecture Maps

Explicit layer and boundary maps for SSQ-LR, HORN, and HBSM real trace packets.
This artifact is config-only: it does not contain activations, SSM state, quality metrics, or GPU evidence.

| Model | Layers | Boundaries | Direction counts | Config hash |
|---|---:|---:|---|---|
| ibm-granite-4.0-h-small | 40 | 8 | `{"attention->ssm": 4, "ssm->attention": 4}` | `8616e9f0b30e` |
| ibm-granite-4.0-h-tiny | 40 | 8 | `{"attention->ssm": 4, "ssm->attention": 4}` | `bda8fd574ace` |
| qwen3-next-80b-a3b-instruct | 48 | 23 | `{"attention->ssm": 11, "ssm->attention": 12}` | `2d483c7cabad` |

Rows are also written to `raw_rows.jsonl` with `layer`, `boundary`,
`non_boundary_control`, and `permuted_direction_control` row types.

## Claim Boundary

These maps can validate boundary IDs and architecture provenance in future real trace packets.
They cannot promote SSQ-LR, HORN, or HBSM without measured model states or activations.
