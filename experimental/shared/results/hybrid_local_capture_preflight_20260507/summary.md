# Hybrid Local Capture Preflight

Decision: `LOCAL_CAPTURE_READY_NOT_EVIDENCE`

This is not model evidence and cannot promote SSQ-LR, HORN, or HBSM.

Next step: Run the first real SSQ-LR/HORN/HBSM tensor capture from the frozen manifest.

| Model | Projects | Cached weights | Transformers class | Est. GB | Decision | Blockers |
|---|---|---|---|---:|---|---|
| ibm-granite-4.0-h-small | hbsm, horn, ssq_lr | no | yes | 59.99 | `LOCAL_CAPTURE_BLOCKED_MODEL_CACHE_NOT_EVIDENCE` | no local cached model weights found; estimated weights 59.99 GB exceed Mac capture budget 24.00 GB |
| ibm-granite-4.0-h-tiny | hbsm, horn, ssq_lr | yes | yes | 12.93 | `LOCAL_CAPTURE_READY_NOT_EVIDENCE` | none |
| qwen3-next-80b-a3b-instruct | hbsm, horn, ssq_lr | no | yes | 151.49 | `LOCAL_CAPTURE_BLOCKED_MODEL_CACHE_NOT_EVIDENCE` | no local cached model weights found; estimated weights 151.49 GB exceed Mac capture budget 24.00 GB |

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

`mamba_ssm` and `vllm` are recorded as optional local/GPU runtime packages here;
they are not hard blockers when cached configs map to native `transformers` hybrid classes.
