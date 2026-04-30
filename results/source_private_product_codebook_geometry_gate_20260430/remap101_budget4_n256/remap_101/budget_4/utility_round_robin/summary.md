# Product-Codebook Geometry Variant

- variant: `utility_round_robin`
- remap seed: `101`
- budget bytes: `4`
- source packet pass: `True`
- source accuracy: `0.574`
- target accuracy: `0.250`
- best control: `random_same_byte` at `0.297`
- source minus canonical: `-0.008`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0015 |
| source | 0.574 | 147/256 | 4.00 | 8.2753 |
| label_shuffled_ridge | 0.215 | 55/256 | 4.00 | 7.7382 |
| constrained_shuffled_source | 0.117 | 30/256 | 4.00 | 8.3394 |
| answer_masked_source | 0.254 | 65/256 | 4.00 | 8.1157 |
| permuted_codes | 0.227 | 58/256 | 4.00 | 8.5208 |
| random_same_byte | 0.297 | 76/256 | 4.00 | 0.0880 |
