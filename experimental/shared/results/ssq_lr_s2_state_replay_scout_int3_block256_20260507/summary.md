# SSQ-LR S2 State Replay Scout

Decision: `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a resource-limited local continuation replay. It cannot promote S2.

- Prompts: `4`
- Rows: `44`
- Contract gate status: `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`
- Selected recipe: `int3_primary_state_block_scaled`
- Selected memory reduction: `5.224x`
- Selected accuracy delta high: `0.000000`

| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |
|---|---:|---:|---:|---:|
| bf16_noop | 0.000000 | 0.000000 | 0.000000 | 1.000x |
| candidate_recipe | 0.005509 | 0.135231 | 0.000000 | 1.984x |
| fp8_state | -0.026609 | 0.096224 | 0.000000 | 2.000x |
| int8_state | 0.005430 | 0.012888 | 0.000000 | 1.984x |
| mxfp4_state | 0.014148 | 0.060450 | 0.000000 | 3.938x |
| random_same_l2 | -0.140258 | 0.844915 | 0.333333 | 3.938x |
| same_byte_uniform | -0.001525 | 0.058176 | 0.000000 | 3.938x |
| shuffled_scales | 2.074598 | 3.845182 | 1.000000 | 3.938x |
