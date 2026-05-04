# HellaSwag Multi-Signal Packet Frontier Gate

- created UTC: `2026-05-04T22:22:51.881618+00:00`
- pass gate: `False`
- eval rows: `768`

## Headline

- fixed-hybrid accuracy: `0.467448`
- multi-signal selector accuracy: `0.455729`
- multi-signal delta vs fixed: `-0.011719`
- multi-signal CI95 low vs fixed: `-0.023470`
- selected overrides: `30`
- source top1/top2 oracle accuracy: `0.694010`
- best destructive control: `field_shuffle_multisignal_control` at `0.430990`

## Method Rows

| Method | Accuracy | Baseline | Delta | CI95 Low | Helps | Harms | Overrides | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_or_source_top1_top2_oracle_diagnostic | `0.694010` | `0.467448` | `0.226562` | `0.196615` | `174` | `0` | `174` | `0` |
| multisignal_packet_frontier | `0.455729` | `0.467448` | `-0.011719` | `-0.023470` | `6` | `15` | `30` | `5` |
| candidate_only | `0.455729` | `0.467448` | `-0.011719` | `-0.022135` | `5` | `14` | `26` | `4` |
| field_shuffle_multisignal_control | `0.430990` | `0.467448` | `-0.036458` | `-0.053385` | `10` | `38` | `69` | `5` |
| source_top1_choice_control | `0.411458` | `0.467448` | `-0.055990` | `-0.079427` | `22` | `65` | `134` | `4` |
| target_derived_packet_multisignal_control | `0.300781` | `0.467448` | `-0.166667` | `-0.210938` | `75` | `203` | `411` | `0` |
| source_row_shuffle_multisignal_control | `0.285156` | `0.467448` | `-0.182292` | `-0.226562` | `101` | `241` | `502` | `5` |
| zero_source_multisignal_control | `0.272135` | `0.467448` | `-0.195312` | `-0.239583` | `101` | `251` | `535` | `0` |
| source_top2_choice_control | `0.264323` | `0.467448` | `-0.203125` | `-0.257812` | `152` | `308` | `673` | `4` |
| target_only | `0.263021` | `0.467448` | `-0.204427` | `-0.255208` | `105` | `262` | `551` | `0` |
| random_same_byte_multisignal_control | `0.259115` | `0.467448` | `-0.208333` | `-0.251302` | `85` | `245` | `496` | `5` |
| candidate_roll_multisignal_control | `0.221354` | `0.467448` | `-0.246094` | `-0.294303` | `123` | `312` | `667` | `5` |

## Interpretation

This gate tests the strongest no-new-inference source packet available after the top1/top2 frontier failed. A failure means cached hidden/score/vote source packet policies still do not expose a stable repair frontier beyond source-choice, routing, and destructive packet controls.
