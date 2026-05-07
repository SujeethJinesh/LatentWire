# Hybrid Manifest Local Capture Runner

Decision: `RESOURCE_LIMITED_CAPTURE_PACKETS_WRITTEN_NOT_PROMOTABLE`

This is a resource-limited local capture packet. It cannot promote SSQ-LR, HORN, or HBSM.

- Model: `ibm-granite/granite-4.0-h-tiny`
- Prompts: `hrs1b_0001, hrs1b_0002, hrs1b_0003, hrs1b_0004, hrs1b_0005, hrs1b_0006, hrs1b_0007, hrs1b_0008, hrs1b_0009, hrs1b_0010, hrs1b_0011, hrs1b_0012`
- Input tokens: `8`
- Load seconds: `3.66`
- Forward seconds: `86.65`

| Project | Checker | Gate packet |
|---|---|---|
| `ssq_lr` | `PASS` | `experimental/shared/results/ssq_lr_s1b_holdout_tensor_capture_20260507/ssq_lr_gate_packet` |
