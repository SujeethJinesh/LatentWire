# Source-Private Anti-Lookup Label-Blind Summary

- pass gate: `True`
- scale rung: `strict-small anti-lookup stress`

## Headline

- rows: `4`
- collapse_pass_rows: `4`
- max_opaque_minus_target: `0.0`
- max_opaque_ci95_high_vs_target: `0.0`
- max_opaque_strict_ci95_high_vs_target: `0.0`
- min_diagnostic_table_positive_lift: `0.42500000000000004`
- min_matched_packet_valid_rate: `1.0`
- all_exact_id_parity: `True`

## Rows

| Surface | Collapse pass | Target | Matched packet | Max opaque | Opaque-target | Max opaque CI high | Positive diagnostic-table lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| core n8 label_blind | `True` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.425 |
| holdout n8 label_blind | `True` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.438 |
| core n32 label_blind | `True` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.425 |
| holdout n32 label_blind | `True` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.438 |

## Interpretation

When candidate repair-key metadata and original labels are hidden, opaque diagnostic packets and text relays collapse to target accuracy while the diagnostic-table endpoint rows remain strongly positive. This weakens the leakage concern that hidden labels alone explain the positive endpoint row, but it also confirms the current method requires a public side-information table. It is not protocol-free semantic transfer. Bootstrap upper bounds here are one-sided collapse diagnostics, not positive-method CIs.

## Next Gate

Run n=160 core+holdout label-blind stress with paired uncertainty, then implement a learned/shared-dictionary receiver if the goal is a protocol-free or less table-shaped method.
