# Source-Private PQ Control-Regularized Receiver

- pass gate: `False`
- examples: `256`
- remap seed: `101`
- variant: `utility_protected_hadamard`
- learned source accuracy: `0.250`
- deterministic L2 source accuracy: `0.270`
- target accuracy: `0.250`
- best learned control: `label_shuffled_ridge` at `0.250`
- learned minus best control: `0.000`

| Condition | Learned acc | L2 acc | Mean bytes | p50 ms |
|---|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1310 |
| source | 0.250 | 0.270 | 4.00 | 1.1359 |
| label_shuffled_ridge | 0.250 | 0.203 | 4.00 | 1.1350 |
| constrained_shuffled_source | 0.250 | 0.227 | 4.00 | 1.1354 |
| answer_masked_source | 0.250 | 0.242 | 4.00 | 1.1343 |
| permuted_codes | 0.250 | 0.270 | 4.00 | 1.1339 |
| random_same_byte | 0.250 | 0.246 | 4.00 | 0.7989 |
| deranged_public_table | 0.250 | 0.195 | 4.00 | 1.1602 |

Pass rule: Pass requires exact ID parity; learned source >= target+0.15; learned source >= best destructive control+0.15; all controls including deranged public table <= target+0.06; and learned source within 0.08 accuracy of deterministic PQ L2 decoding.
