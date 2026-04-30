# Source-Private Masked Consistency Receiver Label-Blind Stress

- pass gate: `True`
- reference full n256 pass: `True`
- opaque slot collapse: `True`
- blinded controls clean: `True`
- all disjoint train/eval: `True`
- min reference full lift vs target: `0.6640625`
- max opaque slot lift vs target: `0.01171875`
- max opaque slot Hamming lift vs target: `0.0234375`
- max blinded lift vs target: `0.74609375`

| Run | n | view | remap | train/eval overlap | pass | learned | Hamming | target | lift | CI high | best control | max control delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `full_seed29_30` | 256 | `full` |  | 0 | `True` | 0.914 | 0.930 | 0.250 | 0.664 | 0.727 | `zero_source` 0.250 | 0.000 |
| `full_seed31_32` | 256 | `full` |  | 0 | `True` | 0.957 | 0.988 | 0.250 | 0.707 | 0.766 | `zero_source` 0.250 | 0.000 |
| `semantic_seed29_30` | 256 | `semantic` |  | 0 | `True` | 0.996 | 0.156 | 0.250 | 0.746 | 0.801 | `zero_source` 0.250 | 0.000 |
| `slot_remap901_seed29_30` | 256 | `slot` | 901 | 0 | `False` | 0.234 | 0.180 | 0.250 | -0.016 | 0.027 | `shuffled_source` 0.254 | 0.004 |
| `slot_remap907_seed31_32` | 256 | `slot` | 907 | 0 | `False` | 0.262 | 0.273 | 0.250 | 0.012 | 0.066 | `wrong_projection_source` 0.266 | 0.016 |

Pass requires existing full-view n256 receiver runs to pass; every opaque slot-view n256 run with per-example remapped candidate order to collapse to target-only within +0.05 for learned and deterministic Hamming decoders with paired CI95 high <= +0.10; all decisive rows to have zero train/eval ID overlap; all destructive controls to stay within target+0.05; and exact-ID parity for all rows. no_diag/semantic rows are diagnostic: if they remain high, the receiver can use public semantic candidate side information; if they collapse, the normal result is diagnostic-key dependent.

This gate distinguishes source-private packet communication with decoder side information from opaque candidate-index lookup. The headline claim is strengthened when normal full-view packets pass while per-example remapped slot-only candidate views collapse.
