# Hybrid Trace Collection Plan

This artifact enumerates the next admissible Mac/local trace captures for SSQ-LR, HORN, and HBSM.
It is not model evidence; it is an execution checklist for producing real packets.

| Project | Planned rows | Trace-plan hash |
|---|---:|---|
| `hbsm` | 2304 | `sha256:015e28d426aa...` |
| `horn` | 1404 | `sha256:a2df7d6485d3...` |
| `ssq_lr` | 5184 | `sha256:a05dab6ad3b8...` |

## Use

Use these JSONL files to populate `ssq_lr_entries`, `horn_entries`, or HBSM row packets,
then run `experimental.shared.hybrid_trace_packet_builder` and `check_gate_packet --mode real`.

## Claim Boundary

A complete trace plan only reduces operational ambiguity. It cannot promote any gate.
