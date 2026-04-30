# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `slot`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.250`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.293`
- unquantized predicted accuracy: `0.270`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0017 |
| source | 0.250 | 4.00 | 0.3729 |
| label_shuffled_encoder | 0.293 | 4.00 | 0.3675 |
| constrained_shuffled_source | 0.285 | 4.00 | 0.3745 |
| answer_masked_source | 0.273 | 4.00 | 0.3645 |
| permuted_codes | 0.258 | 4.00 | 0.3667 |
| random_same_byte | 0.242 | 4.00 | 0.1382 |
| deranged_public_basis | 0.238 | 4.00 | 0.3908 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3676 |

Payload uniqueness: `{'unique_payloads': 227, 'unique_payload_ratio': 0.88671875, 'max_payload_frequency': 3, 'reused_payload_examples': 56, 'reused_payload_accuracy': 0.30357142857142855}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
