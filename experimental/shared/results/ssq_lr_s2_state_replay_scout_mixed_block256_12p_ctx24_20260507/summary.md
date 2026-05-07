# SSQ-LR S2 State Replay Scout

Decision: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `12`
- Rows: `156`
- Contract gate status: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `mixed_int3_mxfp4_low_error_25pct`
- Selected memory reduction: `4.192x`
- Selected accuracy delta high: `0.066667`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | -0.009715 | 0.121775 | 0.133333 | 1.984x |
| fp8_state | 0.001912 | 0.084453 | 0.066667 | 2.000x |
| int8_state | -0.001311 | 0.094509 | 0.066667 | 1.984x |
| mxfp4_state | -0.016878 | 0.090086 | 0.066667 | 3.938x |
| random_same_l2 | 0.114178 | 0.595237 | 0.533333 | 3.938x |
| same_byte_uniform | -0.002913 | 0.133810 | 0.066667 | 3.938x |
| shuffled_scales | 3.929541 | 10.486804 | 1.000000 | 3.938x |
