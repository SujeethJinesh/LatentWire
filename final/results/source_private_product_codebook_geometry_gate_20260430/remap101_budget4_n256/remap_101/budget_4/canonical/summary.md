# Product-Codebook Geometry Variant

- variant: `canonical`
- remap seed: `101`
- budget bytes: `4`
- source packet pass: `True`
- source accuracy: `0.582`
- target accuracy: `0.250`
- best control: `answer_masked_source` at `0.254`
- source minus canonical: `0.000`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0014 |
| source | 0.582 | 149/256 | 4.00 | 6.7755 |
| label_shuffled_ridge | 0.211 | 54/256 | 4.00 | 7.5296 |
| constrained_shuffled_source | 0.113 | 29/256 | 4.00 | 6.8388 |
| answer_masked_source | 0.254 | 65/256 | 4.00 | 6.6159 |
| permuted_codes | 0.199 | 51/256 | 4.00 | 6.2309 |
| random_same_byte | 0.246 | 63/256 | 4.00 | 0.0858 |
