# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `semantic`
- conditioning mode: `public_zscore`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.336`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.375`
- unquantized predicted accuracy: `0.500`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0016 |
| source | 0.336 | 4.00 | 1.3956 |
| label_shuffled_encoder | 0.324 | 4.00 | 1.2504 |
| constrained_shuffled_source | 0.246 | 4.00 | 1.2635 |
| same_answer_slot_wrong_row_source | 0.320 | 4.00 | 1.2678 |
| answer_masked_source | 0.250 | 4.00 | 1.2446 |
| public_condition_only | 0.250 | 4.00 | 1.2443 |
| permuted_codes | 0.375 | 4.00 | 1.2476 |
| random_same_byte | 0.375 | 4.00 | 0.7208 |
| deranged_public_basis | 0.223 | 4.00 | 1.2762 |
| opaque_slot_basis | 0.219 | 4.00 | 0.7524 |

Payload uniqueness: `{'unique_payloads': 187, 'unique_payload_ratio': 0.73046875, 'max_payload_frequency': 15, 'reused_payload_examples': 92, 'reused_payload_accuracy': 0.21739130434782608}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
