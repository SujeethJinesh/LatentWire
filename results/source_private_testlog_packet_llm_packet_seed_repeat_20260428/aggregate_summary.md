# Source-Private Test-Log Packet Seed Repeat Summary

- pass gate: `True`
- interpretation: helper_line repeats pass; full_log/no-helper ablation fails. Treat as protocol-assisted private tool-log packet emission, not unstructured log extraction.

| Prompt mode | Seeds | All pass | Mean matched | Min matched | Mean valid packets | Mean lift vs no-source | Mean lift vs controls |
|---|---|---|---:|---:|---:|---:|---:|
| full_log | [29, 30] | `False` | 0.344 | 0.344 | 0.163 | 0.094 | 0.094 |
| helper_line | [29, 30] | `True` | 0.938 | 0.938 | 0.919 | 0.688 | 0.688 |

## Runs

| Run | Mode | Seed | Pass | Matched | Target-only | Best control | Valid packets | p50 latency ms |
|---|---|---:|---|---:|---:|---:|---:|---:|
| helper_seed29 | helper_line | 29 | `True` | 0.938 | 0.250 | 0.250 | 0.919 | 167.09 |
| helper_seed30 | helper_line | 30 | `True` | 0.938 | 0.250 | 0.250 | 0.919 | 164.70 |
| full_log_seed29 | full_log | 29 | `False` | 0.344 | 0.250 | 0.250 | 0.163 | 132.97 |
| full_log_seed30 | full_log | 30 | `False` | 0.344 | 0.250 | 0.250 | 0.163 | 129.01 |

Next gate: keep helper-line as protocol-assisted handoff, add cross-model confirmation, and move to hidden-test/code-repair logs before treating this as ICLR-ready.
