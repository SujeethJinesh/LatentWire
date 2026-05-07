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
| candidate_recipe | -0.001397 | 0.045054 | 0.066667 | 1.984x |
| fp8_state | -0.000952 | 0.026676 | 0.066667 | 2.000x |
| int8_state | -0.000198 | 0.008338 | 0.000000 | 1.984x |
| mxfp4_state | 0.000524 | 0.025805 | 0.066667 | 3.938x |
| random_same_l2 | -0.006516 | 0.151514 | 0.133333 | 3.938x |
| same_byte_uniform | -0.009667 | 0.058044 | 0.066667 | 3.938x |
| shuffled_scales | 0.342914 | 2.232051 | 0.533333 | 3.938x |
