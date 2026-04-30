# Source-Private PQ Control-Regularized Receiver

- pass gate: `True`
- examples: `500`
- remap seed: `103`
- variant: `utility_protected_hadamard`
- learned source accuracy: `0.504`
- deterministic L2 source accuracy: `0.504`
- target accuracy: `0.250`
- best learned control: `random_same_byte` at `0.298`
- learned minus best control: `0.206`

| Condition | Learned acc | L2 acc | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1310 |
| source | 0.504 | 0.504 | 4.00 | 1.2395 |
| label_shuffled_ridge | 0.266 | 0.266 | 4.00 | 1.2396 |
| constrained_shuffled_source | 0.172 | 0.172 | 4.00 | 1.2427 |
| answer_masked_source | 0.244 | 0.244 | 4.00 | 1.2369 |
| permuted_codes | 0.224 | 0.224 | 4.00 | 1.2375 |
| random_same_byte | 0.298 | 0.298 | 4.00 | 0.8839 |
| deranged_public_table | 0.158 | 0.158 | 4.00 | 1.2624 |

Pass rule: Pass requires exact ID parity; learned source >= target+0.15; learned source >= best destructive control+0.15; all controls including deranged public table <= target+0.06; and learned source within 0.08 accuracy of deterministic PQ L2 decoding.
