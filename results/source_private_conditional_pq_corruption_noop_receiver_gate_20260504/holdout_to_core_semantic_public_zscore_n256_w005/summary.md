# Conditional PQ Corruption-to-Noop Receiver Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- train/eval families: `holdout->core`
- basis view: `semantic`
- conditioning mode: `public_zscore`
- budget bytes: `4`
- source accuracy: `0.254`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.254`
- source minus best control: `0.000`
- CI95 low vs best control: `-0.012`
- helps/harms: `1/0`
- unquantized predicted accuracy: `0.254`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.7120 |
| source | 0.254 | 4.00 | 1.3425 |
| label_shuffled_encoder | 0.250 | 4.00 | 1.2397 |
| constrained_shuffled_source | 0.250 | 4.00 | 1.3079 |
| same_answer_slot_wrong_row_source | 0.250 | 4.00 | 1.3023 |
| answer_masked_source | 0.250 | 4.00 | 1.2323 |
| public_condition_only | 0.250 | 4.00 | 1.2327 |
| permuted_codes | 0.254 | 4.00 | 1.2337 |
| random_same_byte | 0.254 | 4.00 | 0.7088 |
| deranged_public_basis | 0.230 | 4.00 | 1.2676 |
| candidate_roll | 0.234 | 4.00 | 1.2481 |
| opaque_slot_basis | 0.250 | 4.00 | 0.7586 |

Payload uniqueness: `{'unique_payloads': 116, 'unique_payload_ratio': 0.453125, 'max_payload_frequency': 20, 'reused_payload_examples': 188, 'reused_payload_accuracy': 0.2765957446808511}`

Pass rule: Pass requires disjoint train/eval IDs, source >= target+0.05, source >= best destructive/shortcut control+0.10, and paired CI95 low versus best control > 0.
