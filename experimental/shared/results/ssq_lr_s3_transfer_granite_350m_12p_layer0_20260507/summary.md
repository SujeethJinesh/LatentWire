# SSQ-LR S2 State Replay Scout

Decision: `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `12`
- Rows: `156`
- Contract gate status: `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `int3_primary_state_block_scaled`
- Selected memory reduction: `5.224x`
- Selected accuracy delta high: `0.000000`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | 0.000995 | 0.030253 | 0.066667 | 1.984x |
| fp8_state | -0.000353 | 0.030253 | 0.052632 | 2.000x |
| int8_state | -0.000098 | 0.004335 | 0.000000 | 1.984x |
| mxfp4_state | 0.002416 | 0.014099 | 0.000000 | 3.938x |
| random_same_l2 | 0.064059 | 0.154032 | 0.285714 | 3.938x |
| same_byte_uniform | 0.002416 | 0.014099 | 0.000000 | 3.938x |
| shuffled_scales | 2.514193 | 10.925858 | 0.947368 | 3.938x |
