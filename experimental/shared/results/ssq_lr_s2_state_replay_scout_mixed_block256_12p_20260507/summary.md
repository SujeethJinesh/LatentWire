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
| candidate_recipe | -0.005997 | 0.154439 | 0.333333 | 1.984x |
| fp8_state | -0.025476 | 0.154439 | 0.333333 | 2.000x |
| int8_state | -0.003337 | 0.041283 | 0.000000 | 1.984x |
| mxfp4_state | -0.003913 | 0.060450 | 0.000000 | 3.938x |
| random_same_l2 | 0.038987 | 0.938059 | 0.666667 | 3.938x |
| same_byte_uniform | -0.020444 | 0.173317 | 0.000000 | 3.938x |
| shuffled_scales | 2.446765 | 9.632586 | 1.000000 | 3.938x |
