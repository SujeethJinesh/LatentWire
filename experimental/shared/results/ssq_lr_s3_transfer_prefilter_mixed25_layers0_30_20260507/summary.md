# SSQ-LR S3 Transfer Prefilter

Decision: `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER`

This packet is schema-valid S3 preflight evidence, not a cross-model transfer pass.

- Transfer model count: `1`
- Passing model count: `1`
- Complete local transfer candidates: `1`
- Frozen recipe hash: `sha256:df4c3f234306c6cc98c07073ab21a88e67a186ecca62436d49507a95d62bdbc1`
- Max accuracy delta: `0.000000`
- Max CI high: `0.000000`
- Max NLL delta: `0.050439`

| Model | Cache present | Complete weights | Role |
|---|---:|---:|---|
| ibm-granite/granite-4.0-h-tiny | True | True | transfer_candidate |
| ibm-granite/granite-4.0-h-micro | False | False | blocked_missing_weights |
| ibm-granite/granite-4.0-h-350m | False | False | blocked_missing_weights |
| ibm-granite/granite-4.0-h-1b | False | False | blocked_missing_weights |
| ibm-granite/granite-4.0-h-small | True | False | blocked_missing_weights |
| ibm-granite/granite-4.0-h-small-FP8 | True | False | blocked_missing_weights |
| Qwen/Qwen3-Next-80B-A3B-Instruct | True | False | blocked_missing_weights |
