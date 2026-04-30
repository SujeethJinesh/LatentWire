# Source-Private Conditional PQ Innovation Gate

- pass gate: `True`
- examples: `500`
- train/eval ID overlap: `0`
- basis view: `anchor_relative`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.998`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.256`
- unquantized predicted accuracy: `0.822`
- target innovation oracle accuracy: `0.808`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0015 |
| source | 0.998 | 4.00 | 0.6819 |
| label_shuffled_encoder | 0.254 | 4.00 | 0.4771 |
| constrained_shuffled_source | 0.254 | 4.00 | 0.5189 |
| answer_masked_source | 0.250 | 4.00 | 0.4726 |
| permuted_codes | 0.256 | 4.00 | 0.4736 |
| random_same_byte | 0.256 | 4.00 | 0.3970 |
| deranged_public_basis | 0.000 | 4.00 | 0.5003 |
| opaque_slot_basis | 0.250 | 4.00 | 0.1894 |

Payload uniqueness: `{'unique_payloads': 470, 'unique_payload_ratio': 0.94, 'max_payload_frequency': 5, 'reused_payload_examples': 52, 'reused_payload_accuracy': 1.0}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
