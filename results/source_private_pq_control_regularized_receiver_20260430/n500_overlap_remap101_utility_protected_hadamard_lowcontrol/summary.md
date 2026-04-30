# Source-Private PQ Control-Regularized Receiver

- pass gate: `True`
- examples: `500`
- remap seed: `101`
- variant: `utility_protected_hadamard`
- learned source accuracy: `0.504`
- deterministic L2 source accuracy: `0.504`
- target accuracy: `0.250`
- best learned control: `answer_masked_source` at `0.268`
- learned minus best control: `0.236`

| Condition | Learned acc | L2 acc | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1308 |
| source | 0.504 | 0.504 | 4.00 | 0.8734 |
| label_shuffled_ridge | 0.240 | 0.240 | 4.00 | 0.8721 |
| constrained_shuffled_source | 0.178 | 0.178 | 4.00 | 0.8755 |
| answer_masked_source | 0.268 | 0.268 | 4.00 | 0.8669 |
| permuted_codes | 0.256 | 0.256 | 4.00 | 0.8718 |
| random_same_byte | 0.250 | 0.250 | 4.00 | 0.5844 |
| deranged_public_table | 0.180 | 0.180 | 4.00 | 0.9064 |

Pass rule: Pass requires exact ID parity; learned source >= target+0.15; learned source >= best destructive control+0.15; all controls including deranged public table <= target+0.06; and learned source within 0.08 accuracy of deterministic PQ L2 decoding.
