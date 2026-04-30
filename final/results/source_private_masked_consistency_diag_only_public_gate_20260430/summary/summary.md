# Source-Private Masked Consistency Public-Separation Gate

- pass gate: `False`
- rows: `2`
- passed rows: `0`
- min learned minus target: `0.05199999999999999`
- min learned minus best control: `0.05199999999999999`
- max public minus target: `-0.07200000000000001`
- max public explained fraction: `0.0`
- all same eval hash: `True`
- all public near target: `True`

| Packet run | Public run | n | view | learned | target | best control | packet lift | public | public lift | public frac | pass |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `results/source_private_masked_consistency_diag_only_public_gate_20260430/n500_eval29_train30_diag_only_packet` | `results/source_private_diag_only_public_ablation_20260430/n500_seed29_diag_only_public_same_eval` | 500 | `diag_only` | 0.336 | 0.250 | 0.252 | 0.086 | 0.178 | -0.072 | 0.000 | `False` |
| `results/source_private_masked_consistency_diag_only_public_gate_20260430/n500_eval31_train32_diag_only_packet` | `results/source_private_diag_only_public_ablation_20260430/n500_seed31_diag_only_public_same_eval` | 500 | `diag_only` | 0.302 | 0.250 | 0.250 | 0.052 | 0.142 | -0.108 | 0.000 | `False` |

Pass requires exact ID parity, same eval IDs between packet and public-only rows, disjoint train/eval IDs, matched learned packet accuracy >= target+0.15 and >= best destructive control+0.15, paired CI95 lower bounds > +0.10 vs target and best control, destructive controls within target+0.05, public-only accuracy within target+0.05, and public-only lift explaining less than the configured fraction of packet lift.

This aggregate decides whether a learned masked-consistency receiver has source-packet lift that is not explained by a separately trained public-only receiver on the same eval IDs.
