# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `shared_text`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.285`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.340`
- unquantized predicted accuracy: `0.426`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0016 |
| source | 0.285 | 4.00 | 0.8325 |
| label_shuffled_encoder | 0.340 | 4.00 | 0.2931 |
| constrained_shuffled_source | 0.250 | 4.00 | 0.3031 |
| answer_masked_source | 0.250 | 4.00 | 0.2865 |
| permuted_codes | 0.250 | 4.00 | 0.2891 |
| random_same_byte | 0.250 | 4.00 | 0.0965 |
| deranged_public_basis | 0.234 | 4.00 | 0.3148 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3345 |

Payload uniqueness: `{'unique_payloads': 170, 'unique_payload_ratio': 0.6640625, 'max_payload_frequency': 20, 'reused_payload_examples': 114, 'reused_payload_accuracy': 0.2982456140350877}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
