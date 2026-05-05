# ARC Candidate-Local Source-Evidence Repair Preflight

This is an implementation gate, not a paper-positive result. It preserves candidate-local source hidden evidence after the row-level token-pool readout failed.

## Headline

- pass_gate: `False`
- matched accuracy: `0.3125`
- best strict control: `packet_only_source_index` at `0.4375`
- matched minus best control: `-0.1250`

## Condition Metrics

| condition | accuracy | mean margin | matched delta CI |
|---|---:|---:|---:|
| atom_shuffle_control | 0.3125 | -0.1510 | +0.0000 [-0.1875, +0.1875] |
| candidate_roll_control | 0.3750 | -0.0928 | -0.0625 [-0.4375, +0.3750] |
| candidate_source_roll_control | 0.3750 | -0.0904 | -0.0625 [-0.1875, +0.0000] |
| coefficient_shuffle_control | 0.2500 | -2.1948 | +0.0625 [-0.3125, +0.4375] |
| matched_candidate_local_source_repair | 0.3125 | -0.1061 | - |
| packet_only_source_index | 0.4375 | -0.1250 | -0.1250 [-0.5000, +0.2500] |
| public_candidate_readout | 0.3125 | -0.1359 | +0.0000 [-0.2500, +0.2500] |
| same_byte_visible_text | 0.3750 | -0.1951 | -0.0625 [-0.4375, +0.3125] |
| same_source_choice_wrong_row_control | 0.3125 | -0.1388 | +0.0000 [-0.1875, +0.1875] |
| source_only_readout | 0.2500 | -0.1307 | +0.0625 [+0.0000, +0.1875] |
| source_rank_control | 0.4375 | -0.2031 | -0.1250 [-0.5000, +0.2500] |
| source_row_shuffle_control | 0.3125 | -0.0945 | +0.0000 [-0.1875, +0.1875] |
| source_score_control | 0.4375 | -0.5950 | -0.1250 [-0.5000, +0.2500] |
| target_only | 0.2500 | -0.1784 | +0.0625 [-0.2500, +0.3750] |
| wrong_row_source_control | 0.3125 | -0.1188 | +0.0000 [-0.1875, +0.1875] |
| zero_source_control | 0.3750 | -0.1022 | -0.0625 [-0.1875, +0.0000] |

## Systems Diagnostic

- dense candidate source bytes per row: `14336.0`
- hypothetical top-k sparse proxy bytes per row: `72.00`
- This run does not claim a final low-byte packet; it probes whether candidate-local source signal is worth compressing.
