# Hybrid Local Capture Preflight

Decision: `LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE`

This is not model evidence and cannot promote SSQ-LR, HORN, or HBSM.

Next step: Install the missing runtime packages in the repo-local venv, then rerun this preflight.

| Model | Projects | Cached weights | Est. GB | Decision | Blockers |
|---|---|---|---:|---|---|
| ibm-granite-4.0-h-small | hbsm, horn, ssq_lr | no | 59.99 | `LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE` | no local cached model weights found; estimated weights 59.99 GB exceed Mac capture budget 24.00 GB; missing hybrid runtime packages: mamba_ssm |
| ibm-granite-4.0-h-tiny | hbsm, horn, ssq_lr | no | 12.93 | `LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE` | no local cached model weights found; missing hybrid runtime packages: mamba_ssm |
| qwen3-next-80b-a3b-instruct | hbsm, horn, ssq_lr | no | 151.49 | `LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE` | no local cached model weights found; estimated weights 151.49 GB exceed Mac capture budget 24.00 GB; missing hybrid runtime packages: mamba_ssm |

## Environment

- Python: `3.11.6`
- Torch: `2.6.0`
- MPS available: `True`
- Mac weight budget GB: `24.00`

## Package Status

- `torch`: installed
- `transformers`: installed
- `huggingface_hub`: installed
- `mamba_ssm`: missing
- `vllm`: missing
