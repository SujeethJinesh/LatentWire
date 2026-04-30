# Product-Codebook Geometry Variant

- variant: `random_balanced`
- remap seed: `101`
- budget bytes: `4`
- source packet pass: `True`
- source accuracy: `0.586`
- target accuracy: `0.250`
- best control: `random_same_byte` at `0.289`
- source minus canonical: `0.004`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0014 |
| source | 0.586 | 150/256 | 4.00 | 9.1953 |
| label_shuffled_ridge | 0.211 | 54/256 | 4.00 | 8.7772 |
| constrained_shuffled_source | 0.109 | 28/256 | 4.00 | 9.5020 |
| answer_masked_source | 0.254 | 65/256 | 4.00 | 9.4588 |
| permuted_codes | 0.234 | 60/256 | 4.00 | 9.3827 |
| random_same_byte | 0.289 | 74/256 | 4.00 | 0.0869 |
