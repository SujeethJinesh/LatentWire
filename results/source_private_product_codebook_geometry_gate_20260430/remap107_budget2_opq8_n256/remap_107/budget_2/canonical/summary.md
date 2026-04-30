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
| target_only | 0.250 | 64/256 | 0.00 | 0.0014 |
| source | 0.512 | 131/256 | 2.00 | 6.4989 |
| label_shuffled_ridge | 0.242 | 62/256 | 2.00 | 7.4080 |
| constrained_shuffled_source | 0.160 | 41/256 | 2.00 | 7.5128 |
| answer_masked_source | 0.289 | 74/256 | 2.00 | 7.5033 |
| permuted_codes | 0.312 | 80/256 | 2.00 | 7.1721 |
| random_same_byte | 0.277 | 71/256 | 2.00 | 0.0857 |
