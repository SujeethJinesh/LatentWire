# SSQ-LR S2 State Replay Scout

Decision: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `12`
- Rows: `132`
- Contract gate status: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `mxfp4_primary_state_block64`
- Selected memory reduction: `3.938x`
- Selected accuracy delta high: `0.000000`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | -0.007133 | 0.154439 | 0.333333 | 1.984x |
| fp8_state | -0.025476 | 0.154439 | 0.333333 | 2.000x |
| int8_state | -0.003337 | 0.041283 | 0.000000 | 1.984x |
| mxfp4_state | -0.003913 | 0.060450 | 0.000000 | 3.938x |
| random_same_l2 | 0.080117 | 0.844915 | 0.666667 | 3.938x |
| same_byte_uniform | -0.020444 | 0.173317 | 0.000000 | 3.938x |
| shuffled_scales | 2.209480 | 9.865002 | 1.000000 | 3.938x |
