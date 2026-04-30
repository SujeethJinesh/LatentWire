# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `slot`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.262`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.277`
- unquantized predicted accuracy: `0.262`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0023 |
| source | 0.262 | 4.00 | 0.4187 |
| label_shuffled_encoder | 0.277 | 4.00 | 0.3595 |
| constrained_shuffled_source | 0.188 | 4.00 | 0.3714 |
| answer_masked_source | 0.250 | 4.00 | 0.3550 |
| permuted_codes | 0.270 | 4.00 | 0.3581 |
| random_same_byte | 0.266 | 4.00 | 0.1416 |
| deranged_public_basis | 0.250 | 4.00 | 0.3954 |
| opaque_slot_basis | 0.262 | 4.00 | 0.3594 |

Payload uniqueness: `{'unique_payloads': 223, 'unique_payload_ratio': 0.87109375, 'max_payload_frequency': 3, 'reused_payload_examples': 60, 'reused_payload_accuracy': 0.23333333333333334}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
