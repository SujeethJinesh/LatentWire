# Hybrid Manifest Local Capture Runner

Decision: `RESOURCE_LIMITED_CAPTURE_PACKETS_WRITTEN_NOT_PROMOTABLE`

This is a resource-limited local capture packet. It cannot promote SSQ-LR, HORN, or HBSM.

- Model: `ibm-granite/granite-4.0-h-tiny`
- Prompts: `hrsmoke_0001, hrsmoke_0002, hrsmoke_0003, hrsmoke_0004, hrsmoke_0005, hrsmoke_0006, hrsmoke_0007, hrsmoke_0008, hrsmoke_0009, hrsmoke_0010, hrsmoke_0011, hrsmoke_0012`
- Input tokens: `8`
- Load seconds: `5.29`
- Forward seconds: `87.11`

| Project | Checker | Gate packet |
|---|---|---|
| `ssq_lr` | `PASS` | `experimental/shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507/ssq_lr_gate_packet` |
