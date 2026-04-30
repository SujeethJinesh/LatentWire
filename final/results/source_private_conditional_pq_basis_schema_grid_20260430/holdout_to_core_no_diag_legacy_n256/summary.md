# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `no_diag`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.250`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.250`
- unquantized predicted accuracy: `0.523`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0023 |
| source | 0.250 | 4.00 | 0.6701 |
| label_shuffled_encoder | 0.250 | 4.00 | 0.6554 |
| constrained_shuffled_source | 0.250 | 4.00 | 0.6647 |
| answer_masked_source | 0.250 | 4.00 | 0.6479 |
| permuted_codes | 0.250 | 4.00 | 0.6511 |
| random_same_byte | 0.250 | 4.00 | 0.4321 |
| deranged_public_basis | 0.242 | 4.00 | 0.6882 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3597 |

Payload uniqueness: `{'unique_payloads': 219, 'unique_payload_ratio': 0.85546875, 'max_payload_frequency': 4, 'reused_payload_examples': 68, 'reused_payload_accuracy': 0.14705882352941177}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
