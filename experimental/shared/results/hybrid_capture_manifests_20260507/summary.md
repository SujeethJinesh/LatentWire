# Hybrid Capture Manifest Templates

These templates are generated from the frozen trace plans. Fill the marked fields,
write the requested tensors or sensitivity metrics, then run the packet builder.
They are not model evidence and not GPU evidence.

| Project | Model | Entries | Template |
|---|---|---:|---|
| `ssq_lr` | `ibm-granite-4.0-h-small` | 1728 | `experimental/shared/results/hybrid_capture_manifests_20260507/ssq_lr__ibm-granite-4-0-h-small__metadata_template.json` |
| `ssq_lr` | `ibm-granite-4.0-h-tiny` | 1728 | `experimental/shared/results/hybrid_capture_manifests_20260507/ssq_lr__ibm-granite-4-0-h-tiny__metadata_template.json` |
| `ssq_lr` | `qwen3-next-80b-a3b-instruct` | 1728 | `experimental/shared/results/hybrid_capture_manifests_20260507/ssq_lr__qwen3-next-80b-a3b-instruct__metadata_template.json` |
| `horn` | `ibm-granite-4.0-h-small` | 288 | `experimental/shared/results/hybrid_capture_manifests_20260507/horn__ibm-granite-4-0-h-small__metadata_template.json` |
| `horn` | `ibm-granite-4.0-h-tiny` | 288 | `experimental/shared/results/hybrid_capture_manifests_20260507/horn__ibm-granite-4-0-h-tiny__metadata_template.json` |
| `horn` | `qwen3-next-80b-a3b-instruct` | 828 | `experimental/shared/results/hybrid_capture_manifests_20260507/horn__qwen3-next-80b-a3b-instruct__metadata_template.json` |
| `hbsm` | `ibm-granite-4.0-h-small` | 720 | `experimental/shared/results/hybrid_capture_manifests_20260507/hbsm__ibm-granite-4-0-h-small__row_packet_template.json` |
| `hbsm` | `ibm-granite-4.0-h-tiny` | 720 | `experimental/shared/results/hybrid_capture_manifests_20260507/hbsm__ibm-granite-4-0-h-tiny__row_packet_template.json` |
| `hbsm` | `qwen3-next-80b-a3b-instruct` | 864 | `experimental/shared/results/hybrid_capture_manifests_20260507/hbsm__qwen3-next-80b-a3b-instruct__row_packet_template.json` |

Do not pass these templates directly to a packet builder. Builders reject
`_template_only: true` and `TO_FILL_BEFORE_CAPTURE` markers.
