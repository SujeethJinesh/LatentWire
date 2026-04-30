# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `shared_text`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.297`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.277`
- unquantized predicted accuracy: `0.418`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0020 |
| source | 0.297 | 4.00 | 0.8718 |
| label_shuffled_encoder | 0.277 | 4.00 | 0.3347 |
| constrained_shuffled_source | 0.250 | 4.00 | 0.3725 |
| answer_masked_source | 0.250 | 4.00 | 0.3237 |
| permuted_codes | 0.250 | 4.00 | 0.3241 |
| random_same_byte | 0.250 | 4.00 | 0.1037 |
| deranged_public_basis | 0.238 | 4.00 | 0.3595 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3775 |

Payload uniqueness: `{'unique_payloads': 128, 'unique_payload_ratio': 0.5, 'max_payload_frequency': 16, 'reused_payload_examples': 165, 'reused_payload_accuracy': 0.3575757575757576}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
