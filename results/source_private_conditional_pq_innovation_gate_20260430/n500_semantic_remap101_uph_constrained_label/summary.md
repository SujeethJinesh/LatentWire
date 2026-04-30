# Source-Private Conditional PQ Innovation Gate

- pass gate: `True`
- examples: `500`
- train/eval ID overlap: `0`
- basis view: `semantic`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `1.000`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.278`
- unquantized predicted accuracy: `0.836`
- target innovation oracle accuracy: `0.782`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0016 |
| source | 1.000 | 4.00 | 0.7460 |
| label_shuffled_encoder | 0.278 | 4.00 | 0.5375 |
| constrained_shuffled_source | 0.264 | 4.00 | 0.5950 |
| answer_masked_source | 0.250 | 4.00 | 0.5316 |
| permuted_codes | 0.250 | 4.00 | 0.5333 |
| random_same_byte | 0.254 | 4.00 | 0.3862 |
| deranged_public_basis | 0.000 | 4.00 | 0.5625 |
| opaque_slot_basis | 0.250 | 4.00 | 0.2829 |

Payload uniqueness: `{'unique_payloads': 437, 'unique_payload_ratio': 0.874, 'max_payload_frequency': 7, 'reused_payload_examples': 107, 'reused_payload_accuracy': 1.0}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
