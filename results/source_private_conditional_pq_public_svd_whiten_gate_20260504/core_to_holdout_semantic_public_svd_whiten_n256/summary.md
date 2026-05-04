# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `semantic`
- conditioning mode: `public_svd_whiten`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.375`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.613`
- unquantized predicted accuracy: `0.250`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0016 |
| source | 0.375 | 4.00 | 1.5020 |
| label_shuffled_encoder | 0.379 | 4.00 | 1.3447 |
| constrained_shuffled_source | 0.371 | 4.00 | 1.3919 |
| same_answer_slot_wrong_row_source | 0.367 | 4.00 | 1.3923 |
| answer_masked_source | 0.375 | 4.00 | 1.3397 |
| public_condition_only | 0.375 | 4.00 | 1.3369 |
| permuted_codes | 0.613 | 4.00 | 1.3409 |
| random_same_byte | 0.516 | 4.00 | 0.7645 |
| deranged_public_basis | 0.230 | 4.00 | 1.3745 |
| opaque_slot_basis | 0.266 | 4.00 | 0.8677 |

Payload uniqueness: `{'unique_payloads': 21, 'unique_payload_ratio': 0.08203125, 'max_payload_frequency': 32, 'reused_payload_examples': 253, 'reused_payload_accuracy': 0.3794466403162055}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
