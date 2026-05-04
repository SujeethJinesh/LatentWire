# Conditional PQ Corruption-to-Noop Receiver Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- train/eval families: `core->holdout`
- basis view: `semantic`
- conditioning mode: `public_zscore`
- budget bytes: `4`
- source accuracy: `0.246`
- target accuracy: `0.250`
- best control: `permuted_codes` at `0.305`
- source minus best control: `-0.059`
- CI95 low vs best control: `-0.090`
- helps/harms: `0/1`
- unquantized predicted accuracy: `0.500`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.7349 |
| source | 0.246 | 4.00 | 1.4159 |
| label_shuffled_encoder | 0.250 | 4.00 | 1.2499 |
| constrained_shuffled_source | 0.250 | 4.00 | 1.2960 |
| same_answer_slot_wrong_row_source | 0.254 | 4.00 | 1.3038 |
| answer_masked_source | 0.250 | 4.00 | 1.2461 |
| public_condition_only | 0.250 | 4.00 | 1.2451 |
| permuted_codes | 0.305 | 4.00 | 1.2500 |
| random_same_byte | 0.270 | 4.00 | 0.7301 |
| deranged_public_basis | 0.234 | 4.00 | 1.2838 |
| candidate_roll | 0.242 | 4.00 | 1.2626 |
| opaque_slot_basis | 0.266 | 4.00 | 0.7536 |

Payload uniqueness: `{'unique_payloads': 187, 'unique_payload_ratio': 0.73046875, 'max_payload_frequency': 15, 'reused_payload_examples': 92, 'reused_payload_accuracy': 0.09782608695652174}`

Pass rule: Pass requires disjoint train/eval IDs, source >= target+0.05, source >= best destructive/shortcut control+0.10, and paired CI95 low versus best control > 0.
