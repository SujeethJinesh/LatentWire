# ARC Tokenwise Repair Readout Preflight

This is an implementation gate, not a paper-positive result. It tests whether source-token evidence contains held-out repair signal before spending more work on a low-byte packet decoder.

## Headline

- pass_gate: `False`
- matched accuracy: `0.3125`
- best strict control: `packet_only_source_index` at `0.4375`
- matched minus best control: `-0.1250`

## Condition Metrics

| condition | accuracy | mean margin | matched delta CI |
|---|---:|---:|---:|
| atom_shuffle_control | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| candidate_roll_control | 0.0625 | -0.1321 | +0.2500 [+0.0000, +0.5000] |
| coefficient_shuffle_control | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| matched_tokenwise_repair_readout | 0.3125 | -0.1177 | - |
| packet_only_source_index | 0.4375 | -0.1250 | -0.1250 [-0.4375, +0.1875] |
| public_candidate_readout | 0.3125 | -0.1359 | +0.0000 [+0.0000, +0.0000] |
| same_byte_visible_text | 0.3750 | -0.1951 | -0.0625 [-0.4375, +0.3125] |
| same_source_choice_wrong_row_control | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| source_only_readout | 0.1875 | -0.1084 | +0.1250 [-0.1250, +0.3750] |
| source_row_shuffle_control | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| target_only | 0.2500 | -0.1784 | +0.0625 [-0.2500, +0.3750] |
| wrong_row_source_control | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| zero_source_control | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |

## Systems Diagnostic

- dense source feature bytes per row: `57344`
- hypothetical top-k sparse proxy bytes per row: `44.00`
- This run does not claim a final low-byte packet; it only probes whether there is source signal worth compressing.
