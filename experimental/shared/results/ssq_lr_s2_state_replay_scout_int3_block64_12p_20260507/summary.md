# SSQ-LR S2 State Replay Scout

Decision: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `12`
- Rows: `132`
- Contract gate status: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `mxfp4_primary_state_block64`
- Selected memory reduction: `3.765x`
- Selected accuracy delta high: `0.000000`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | -0.008725 | 0.154439 | 0.333333 | 1.939x |
| fp8_state | -0.025476 | 0.154439 | 0.333333 | 2.000x |
| int8_state | -0.009487 | 0.070027 | 0.000000 | 1.939x |
| mxfp4_state | -0.005317 | 0.047962 | 0.000000 | 3.765x |
| random_same_l2 | 0.035459 | 0.872942 | 0.666667 | 3.765x |
| same_byte_uniform | -0.014741 | 0.082419 | 0.000000 | 3.765x |
| shuffled_scales | 1.519637 | 3.634709 | 1.000000 | 3.765x |
