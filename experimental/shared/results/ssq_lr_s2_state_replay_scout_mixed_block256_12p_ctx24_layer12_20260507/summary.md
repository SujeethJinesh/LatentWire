# SSQ-LR S2 State Replay Scout

Decision: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `12`
- Rows: `156`
- Contract gate status: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `fp8_e4m3_primary_state`
- Selected memory reduction: `2.000x`
- Selected accuracy delta high: `0.066667`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | -0.009344 | 0.117276 | 0.066667 | 1.984x |
| fp8_state | -0.006094 | 0.069032 | 0.066667 | 2.000x |
| int8_state | -0.007552 | 0.108296 | 0.066667 | 1.984x |
| mxfp4_state | -0.011640 | 0.105910 | 0.066667 | 3.938x |
| random_same_l2 | 0.112800 | 0.438575 | 0.533333 | 3.938x |
| same_byte_uniform | -0.009667 | 0.058044 | 0.066667 | 3.938x |
| shuffled_scales | 3.225874 | 10.198925 | 1.000000 | 3.938x |
