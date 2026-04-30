# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `shared_text`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.254`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.273`
- unquantized predicted accuracy: `0.410`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0017 |
| source | 0.254 | 4.00 | 0.8173 |
| label_shuffled_encoder | 0.273 | 4.00 | 0.2942 |
| constrained_shuffled_source | 0.250 | 4.00 | 0.3172 |
| answer_masked_source | 0.250 | 4.00 | 0.2874 |
| permuted_codes | 0.250 | 4.00 | 0.2891 |
| random_same_byte | 0.250 | 4.00 | 0.0967 |
| deranged_public_basis | 0.242 | 4.00 | 0.3176 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3355 |

Payload uniqueness: `{'unique_payloads': 133, 'unique_payload_ratio': 0.51953125, 'max_payload_frequency': 19, 'reused_payload_examples': 166, 'reused_payload_accuracy': 0.28313253012048195}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
