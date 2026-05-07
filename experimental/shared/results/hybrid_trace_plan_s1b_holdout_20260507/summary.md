# Hybrid Trace Collection Plan

This artifact enumerates the next admissible Mac/local trace captures for SSQ-LR, HORN, and HBSM.
It is not model evidence; it is an execution checklist for producing real packets.

| Project | Planned rows | Trace-plan hash |
|---|---:|---|
| `hbsm` | 2304 | `sha256:ba532c101b2f...` |
| `horn` | 1404 | `sha256:c32e3572eb33...` |
| `ssq_lr` | 5184 | `sha256:8fefebf0a704...` |

## Use

Use these JSONL files to populate `ssq_lr_entries`, `horn_entries`, or HBSM row packets,
then run `experimental.shared.hybrid_trace_packet_builder` and `check_gate_packet --mode real`.

## Claim Boundary

A complete trace plan only reduces operational ambiguity. It cannot promote any gate.
