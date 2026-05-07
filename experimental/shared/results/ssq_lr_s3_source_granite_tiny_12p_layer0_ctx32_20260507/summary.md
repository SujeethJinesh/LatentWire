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
| candidate_recipe | -0.002329 | 0.123213 | 0.105263 | 1.984x |
| fp8_state | 0.008178 | 0.064114 | 0.105263 | 2.000x |
| int8_state | 0.001902 | 0.024748 | 0.052632 | 1.984x |
| mxfp4_state | -0.006066 | 0.029869 | 0.052632 | 3.938x |
| random_same_l2 | -0.003263 | 0.045022 | 0.052632 | 3.938x |
| same_byte_uniform | -0.006066 | 0.029869 | 0.052632 | 3.938x |
| shuffled_scales | 1.466752 | 9.920473 | 0.842105 | 3.938x |
