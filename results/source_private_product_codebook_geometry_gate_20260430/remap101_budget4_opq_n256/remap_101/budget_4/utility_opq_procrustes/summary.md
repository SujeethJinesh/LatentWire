# Product-Codebook Geometry Variant

- variant: `utility_opq_procrustes`
- remap seed: `101`
- budget bytes: `4`
- source packet pass: `False`
- source accuracy: `0.578`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.316`
- source minus canonical: `-0.004`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0020 |
| source | 0.578 | 148/256 | 4.00 | 50.5401 |
| label_shuffled_ridge | 0.219 | 56/256 | 4.00 | 56.6811 |
| constrained_shuffled_source | 0.109 | 28/256 | 4.00 | 55.4884 |
| answer_masked_source | 0.254 | 65/256 | 4.00 | 56.1272 |
| permuted_codes | 0.316 | 81/256 | 4.00 | 54.5832 |
| random_same_byte | 0.273 | 70/256 | 4.00 | 27.1807 |
