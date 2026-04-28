# Source-Private Test-Log Packet Strict-Small Gate

- strict-small pass: `True`
- passing budgets: `[2, 4, 8, 16, 32]`
- best budget bytes: `2`

| Budget bytes | Pass | Matched | Best no-source | Best control | Matched text | Full log | Full signature |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 |
| 4 | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 |
| 8 | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 |
| 16 | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 |
| 32 | `True` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 |

Pass rule: At least one budget must have matched_testlog_packet - best no-source >= 0.15, all source-destroying controls within +0.02 of no-source, matched-byte structured text within +0.02 of no-source, exact ID parity, and candidate pool recall 1.0.
