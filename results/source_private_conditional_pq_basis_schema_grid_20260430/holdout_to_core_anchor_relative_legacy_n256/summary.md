# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `anchor_relative`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.270`
- target accuracy: `0.250`
- best control: `constrained_shuffled_source` at `0.270`
- unquantized predicted accuracy: `0.445`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0018 |
| source | 0.270 | 4.00 | 0.4951 |
| label_shuffled_encoder | 0.258 | 4.00 | 0.4756 |
| constrained_shuffled_source | 0.270 | 4.00 | 0.4875 |
| answer_masked_source | 0.250 | 4.00 | 0.4705 |
| permuted_codes | 0.254 | 4.00 | 0.4728 |
| random_same_byte | 0.258 | 4.00 | 0.3947 |
| deranged_public_basis | 0.234 | 4.00 | 0.4994 |
| opaque_slot_basis | 0.250 | 4.00 | 0.1930 |

Payload uniqueness: `{'unique_payloads': 242, 'unique_payload_ratio': 0.9453125, 'max_payload_frequency': 2, 'reused_payload_examples': 28, 'reused_payload_accuracy': 0.25}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
