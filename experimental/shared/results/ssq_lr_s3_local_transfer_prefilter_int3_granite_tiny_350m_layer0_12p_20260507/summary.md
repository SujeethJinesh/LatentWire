# SSQ-LR Local S3 Transfer Prefilter

Decision: `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER`
Resource-limited label: `LOCAL_PREFLIGHT_NOT_PROMOTABLE_FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER`

This packet is local two-model no-retuning evidence, not camera-ready cross-model transfer.

- Transfer model count: `2`
- Passing model count: `1`
- Minimum prompt count per model: `12`
- Frozen recipe hash: `sha256:a460bb6aecb7fd8ddde17c52d2c368d239e77a3e711a7fdc08329846c5a59154`
- Max accuracy delta: `0.052632`
- Max CI high: `0.052632`
- Max NLL delta: `0.123213`

| Packet | Role | Models | Prompts | NLL delta abs | NLL CI high |
|---|---|---|---:|---:|---:|
| experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507 | source_model_frozen_recipe_reference | ibm-granite/granite-4.0-h-tiny | 12 | 0.041124 | 0.041124 |
| experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507 | transfer_model_no_retune_local_replay | ibm-granite/granite-4.0-h-350m | 12 | 0.015687 | 0.015687 |
