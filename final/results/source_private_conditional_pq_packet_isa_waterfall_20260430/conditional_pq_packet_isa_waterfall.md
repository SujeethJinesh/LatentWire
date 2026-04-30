# Conditional PQ Packet ISA Waterfall

- pass gate: `True`

## Checks

| Check | Pass |
|---|---:|
| `method_rows_positive` | `True` |
| `ci95_low_positive` | `True` |
| `record_bytes_within_packet_isa` | `True` |
| `transport_under_1us` | `True` |
| `receiver_under_0p25ms` | `True` |
| `receiver_exact` | `True` |
| `private_state_exposure_separated` | `True` |

## Rows

| Row | Acc min | Target | Best ctrl | CI95 low | Record B | Line B/req | DMA B/req | p95 ns | recv p50 ms | Text? | KV? | DMA ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B conditional innovation packet | 1.000 | 0.250 | 0.302 | 0.658 | 5 | 5.0 | 6.0 | 0.688 | 0.01628 | `False` | `False` | 1.00 |
| 4B conditional innovation packet | 0.996 | 0.250 | 0.278 | 0.680 | 7 | 7.0 | 8.0 | 0.661 | 0.01628 | `False` | `False` | 1.33 |
| query-aware private text |  |  |  |  | 14 | 14.0 | 14.0 | 0.692 |  | `True` | `False` | 2.33 |
| full hidden-log relay |  |  |  |  | 370 | 370.0 | 370.0 | 5.652 |  | `True` | `False` | 61.67 |
| QJL 1-bit KV floor |  |  |  |  | 21504 | 21504.0 | 21504.0 | 404.175 |  | `False` | `True` | 3584.00 |
| KIVI/KVQuant 2-bit KV floor |  |  |  |  | 43008 | 43008.0 | 43008.0 | 1169.556 |  | `False` | `True` | 7168.00 |

## Interpretation

This table attaches the conditional-innovation positive rows to the Mac packet-ISA transport/receiver waterfall. It supports a byte-movement and exposure-accounting systems claim, not a measured GPU serving or energy claim.
