# Product-Codebook Geometry Variant

- variant: `opq_procrustes`
- remap seed: `101`
- budget bytes: `4`
- source packet pass: `True`
- source accuracy: `0.578`
- target accuracy: `0.250`
- best control: `random_same_byte` at `0.270`
- source minus canonical: `-0.004`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0020 |
| source | 0.578 | 148/256 | 4.00 | 51.1956 |
| label_shuffled_ridge | 0.219 | 56/256 | 4.00 | 49.1729 |
| constrained_shuffled_source | 0.102 | 26/256 | 4.00 | 53.2032 |
| answer_masked_source | 0.254 | 65/256 | 4.00 | 52.1182 |
| permuted_codes | 0.250 | 64/256 | 4.00 | 50.5130 |
| random_same_byte | 0.270 | 69/256 | 4.00 | 24.2066 |
