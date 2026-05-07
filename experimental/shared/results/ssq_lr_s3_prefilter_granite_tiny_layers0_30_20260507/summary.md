# SSQ-LR S2 State Replay Scout

Decision: `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `12`
- Rows: `156`
- Contract gate status: `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `mixed_int3_mxfp4_low_error_25pct`
- Selected memory reduction: `4.192x`
- Selected accuracy delta high: `0.000000`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | -0.004197 | 0.135033 | 0.105263 | 1.984x |
| fp8_state | 0.010989 | 0.069674 | 0.105263 | 2.000x |
| int8_state | 0.001749 | 0.024537 | 0.052632 | 1.984x |
| mxfp4_state | -0.008353 | 0.032241 | 0.052632 | 3.938x |
| random_same_l2 | 0.020342 | 0.086990 | 0.210526 | 3.938x |
| same_byte_uniform | -0.000383 | 0.047175 | 0.105263 | 3.938x |
| shuffled_scales | 1.252161 | 7.950545 | 0.947368 | 3.938x |
