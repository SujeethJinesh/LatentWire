# Product-Codebook Geometry Variant

- variant: `utility_opq_procrustes`
- remap seed: `107`
- budget bytes: `2`
- source packet pass: `True`
- source accuracy: `0.512`
- target accuracy: `0.250`
- best control: `answer_masked_source` at `0.289`
- source minus canonical: `0.000`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0020 |
| source | 0.512 | 131/256 | 2.00 | 29.0986 |
| label_shuffled_ridge | 0.246 | 63/256 | 2.00 | 31.3562 |
| constrained_shuffled_source | 0.160 | 41/256 | 2.00 | 30.3401 |
| answer_masked_source | 0.289 | 74/256 | 2.00 | 33.1881 |
| permuted_codes | 0.266 | 68/256 | 2.00 | 29.6832 |
| random_same_byte | 0.246 | 63/256 | 2.00 | 14.2763 |
