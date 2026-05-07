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
| candidate_recipe | 0.000551 | 0.040859 | 0.066667 | 1.984x |
| fp8_state | 0.002449 | 0.040859 | 0.066667 | 2.000x |
| int8_state | 0.001163 | 0.008893 | 0.000000 | 1.984x |
| mxfp4_state | -0.000318 | 0.012883 | 0.052632 | 3.938x |
| random_same_l2 | 0.012116 | 0.088053 | 0.052632 | 3.938x |
| same_byte_uniform | 0.002416 | 0.014099 | 0.000000 | 3.938x |
| shuffled_scales | 0.001283 | 0.031292 | 0.105263 | 3.938x |
