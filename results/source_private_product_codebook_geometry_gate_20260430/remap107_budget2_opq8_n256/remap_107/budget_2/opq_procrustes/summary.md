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
| target_only | 0.250 | 64/256 | 0.00 | 0.0019 |
| source | 0.527 | 135/256 | 2.00 | 20.0648 |
| label_shuffled_ridge | 0.242 | 62/256 | 2.00 | 21.9541 |
| constrained_shuffled_source | 0.156 | 40/256 | 2.00 | 21.6417 |
| answer_masked_source | 0.289 | 74/256 | 2.00 | 21.3947 |
| permuted_codes | 0.254 | 65/256 | 2.00 | 20.2491 |
| random_same_byte | 0.277 | 71/256 | 2.00 | 10.1394 |
