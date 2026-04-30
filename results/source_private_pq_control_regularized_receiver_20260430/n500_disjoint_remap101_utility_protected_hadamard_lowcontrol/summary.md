# Source-Private PQ Control-Regularized Receiver

- pass gate: `False`
- examples: `500`
- remap seed: `101`
- variant: `utility_protected_hadamard`
- learned source accuracy: `0.264`
- deterministic L2 source accuracy: `0.264`
- target accuracy: `0.250`
- best learned control: `permuted_codes` at `0.288`
- learned minus best control: `-0.024`

| Condition | Learned acc | L2 acc | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1309 |
| source | 0.264 | 0.264 | 4.00 | 0.9929 |
| label_shuffled_ridge | 0.270 | 0.270 | 4.00 | 0.9915 |
| constrained_shuffled_source | 0.256 | 0.256 | 4.00 | 0.9966 |
| answer_masked_source | 0.258 | 0.258 | 4.00 | 0.9903 |
| permuted_codes | 0.288 | 0.288 | 4.00 | 0.9920 |
| random_same_byte | 0.256 | 0.256 | 4.00 | 0.6873 |
| deranged_public_table | 0.234 | 0.234 | 4.00 | 1.0172 |

Pass rule: Pass requires exact ID parity; learned source >= target+0.15; learned source >= best destructive control+0.15; all controls including deranged public table <= target+0.06; and learned source within 0.08 accuracy of deterministic PQ L2 decoding.
