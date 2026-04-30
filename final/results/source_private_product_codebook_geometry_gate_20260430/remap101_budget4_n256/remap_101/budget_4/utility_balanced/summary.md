# Product-Codebook Geometry Variant

- variant: `utility_balanced`
- remap seed: `101`
- budget bytes: `4`
- source packet pass: `True`
- source accuracy: `0.582`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.289`
- source minus canonical: `0.000`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0014 |
| source | 0.582 | 149/256 | 4.00 | 9.9348 |
| label_shuffled_ridge | 0.211 | 54/256 | 4.00 | 9.1932 |
| constrained_shuffled_source | 0.117 | 30/256 | 4.00 | 9.9146 |
| answer_masked_source | 0.254 | 65/256 | 4.00 | 9.6246 |
| permuted_codes | 0.289 | 74/256 | 4.00 | 10.2285 |
| random_same_byte | 0.273 | 70/256 | 4.00 | 0.0870 |
