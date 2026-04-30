# Product-Codebook Geometry Variant

- variant: `canonical`
- remap seed: `107`
- budget bytes: `2`
- source packet pass: `False`
- source accuracy: `0.512`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.312`
- source minus canonical: `0.000`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0015 |
| source | 0.512 | 131/256 | 2.00 | 11.2346 |
| label_shuffled_ridge | 0.242 | 62/256 | 2.00 | 10.5465 |
| constrained_shuffled_source | 0.160 | 41/256 | 2.00 | 10.8307 |
| answer_masked_source | 0.289 | 74/256 | 2.00 | 10.4694 |
| permuted_codes | 0.312 | 80/256 | 2.00 | 10.3731 |
| random_same_byte | 0.277 | 71/256 | 2.00 | 0.0871 |
