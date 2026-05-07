# SSQ-LR S2 State Replay Scout

Decision: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `12`
- Rows: `156`
- Contract gate status: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `int8_primary_state_block64`
- Selected memory reduction: `1.984x`
- Selected accuracy delta high: `0.000000`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | -0.000038 | 0.034324 | 0.105263 | 1.984x |
| fp8_state | 0.001442 | 0.034324 | 0.066667 | 2.000x |
| int8_state | 0.000245 | 0.008906 | 0.000000 | 1.984x |
| mxfp4_state | 0.000448 | 0.013821 | 0.052632 | 3.938x |
| random_same_l2 | 0.074028 | 0.231179 | 0.285714 | 3.938x |
| same_byte_uniform | 0.002767 | 0.018253 | 0.066667 | 3.938x |
| shuffled_scales | 2.514048 | 10.932342 | 0.947368 | 3.938x |
