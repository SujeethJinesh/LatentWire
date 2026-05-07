# SSQ-LR S2 State Replay Scout

Decision: `RESOURCE_LIMITED_S2_SCOUT_NOT_PROMOTABLE_FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `4`
- Rows: `40`
- Contract gate status: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `mxfp4_primary_state_block64`
- Selected memory reduction: `3.765x`
- Selected accuracy delta high: `0.000000`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | -0.007103 | 0.096224 | 0.000000 | 1.939x |
| fp8_state | -0.026609 | 0.096224 | 0.000000 | 2.000x |
| int8_state | 0.001384 | 0.009059 | 0.000000 | 1.939x |
| mxfp4_state | 0.003916 | 0.028649 | 0.000000 | 3.765x |
| random_same_l2 | -0.259595 | 1.467991 | 0.333333 | 3.765x |
| same_byte_uniform | -0.008037 | 0.046237 | 0.000000 | 3.765x |
| shuffled_scales | 1.544269 | 5.336749 | 1.000000 | 3.765x |
