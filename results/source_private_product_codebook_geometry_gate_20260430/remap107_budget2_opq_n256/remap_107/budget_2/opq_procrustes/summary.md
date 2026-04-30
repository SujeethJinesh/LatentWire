# Product-Codebook Geometry Variant

- variant: `opq_procrustes`
- remap seed: `107`
- budget bytes: `2`
- source packet pass: `True`
- source accuracy: `0.527`
- target accuracy: `0.250`
- best control: `answer_masked_source` at `0.289`
- source minus canonical: `0.016`

| Condition | Accuracy | Correct | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 64/256 | 0.00 | 0.0021 |
| source | 0.527 | 135/256 | 2.00 | 46.1719 |
| label_shuffled_ridge | 0.242 | 62/256 | 2.00 | 49.2177 |
| constrained_shuffled_source | 0.160 | 41/256 | 2.00 | 48.1586 |
| answer_masked_source | 0.289 | 74/256 | 2.00 | 46.8323 |
| permuted_codes | 0.281 | 72/256 | 2.00 | 50.4840 |
| random_same_byte | 0.254 | 65/256 | 2.00 | 24.6655 |
