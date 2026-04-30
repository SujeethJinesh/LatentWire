# Source-Private Conditional PQ Innovation Gate

- pass gate: `True`
- examples: `500`
- train/eval ID overlap: `0`
- basis view: `shared_text`
- variant: `utility_protected_hadamard`
- budget bytes: `2`
- source accuracy: `1.000`
- target accuracy: `0.250`
- best control: `random_same_byte` at `0.288`
- unquantized predicted accuracy: `0.826`
- target innovation oracle accuracy: `0.782`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0015 |
| source | 1.000 | 2.00 | 0.7464 |
| label_shuffled_encoder | 0.272 | 2.00 | 0.2256 |
| constrained_shuffled_source | 0.264 | 2.00 | 0.2295 |
| answer_masked_source | 0.250 | 2.00 | 0.2188 |
| permuted_codes | 0.280 | 2.00 | 0.2207 |
| random_same_byte | 0.288 | 2.00 | 0.0970 |
| deranged_public_basis | 0.000 | 2.00 | 0.2429 |
| opaque_slot_basis | 0.250 | 2.00 | 0.2672 |

Payload uniqueness: `{'unique_payloads': 266, 'unique_payload_ratio': 0.532, 'max_payload_frequency': 15, 'reused_payload_examples': 326, 'reused_payload_accuracy': 1.0}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
