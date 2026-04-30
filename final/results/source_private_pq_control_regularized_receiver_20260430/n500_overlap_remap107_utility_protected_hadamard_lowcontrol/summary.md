# Source-Private PQ Control-Regularized Receiver

- pass gate: `True`
- examples: `500`
- remap seed: `107`
- variant: `utility_protected_hadamard`
- learned source accuracy: `0.516`
- deterministic L2 source accuracy: `0.516`
- target accuracy: `0.250`
- best learned control: `permuted_codes` at `0.262`
- learned minus best control: `0.254`

| Condition | Learned acc | L2 acc | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1309 |
| source | 0.516 | 0.516 | 4.00 | 0.9935 |
| label_shuffled_ridge | 0.240 | 0.240 | 4.00 | 0.9920 |
| constrained_shuffled_source | 0.166 | 0.166 | 4.00 | 0.9947 |
| answer_masked_source | 0.234 | 0.234 | 4.00 | 0.9903 |
| permuted_codes | 0.262 | 0.262 | 4.00 | 0.9911 |
| random_same_byte | 0.258 | 0.258 | 4.00 | 0.6879 |
| deranged_public_table | 0.124 | 0.124 | 4.00 | 1.0177 |

Pass rule: Pass requires exact ID parity; learned source >= target+0.15; learned source >= best destructive control+0.15; all controls including deranged public table <= target+0.06; and learned source within 0.08 accuracy of deterministic PQ L2 decoding.
