# Source-Private Hidden-Repair Packet Strict-Small Gate

- date: `2026-04-29`
- status: promoted strict-small, not yet ICLR-ready
- live branch: explicit source-private tool-trace packet handoff
- scale rung: strict small

## Question

Does the weakened-helper hidden-repair packet protocol survive at `160` frozen
examples and replicate across Qwen3 and Phi-3?

## Setup

The benchmark scales the hidden-repair surface to `160` examples from the same
eight repair families. The target sees public issue text, buggy code, and a
four-candidate repair pool. The source sees a private hidden execution log with
a compact `REPAIR_DIAG` tool-trace field. The primary prompt mode is
`trace_no_hint`, which removes the copied helper line and hint but keeps the
private trace field.

`raw_log_no_trace` is retained as a source-signal destruction control.

## Results

Deterministic packet sweep:

| Budget bytes | Matched | Best no-source | Best control | Matched text | Full log |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 4 | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 8 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 16 | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 32 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |

Model-produced packets:

| Run | Model | Prompt mode | Pass | Matched | Target-only | Best control | Valid packets | Mean bytes | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.794 | 0.250 | 0.256 | 0.762 | 1.52 | 379.06 |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.256 | 1.000 | 2.00 | 431.40 |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.256 | 0.000 | 0.00 | 286.20 |

## Interpretation

This is the first strict-small positive result for the source-private pivot.
The method is explicit and interpretable: a source model transmits a compact
private tool-trace field, and a target uses that packet with candidate-side
metadata. The signal survives source model family changes and disappears when
the trace field is removed.

The claim remains bounded. This is not raw-log repair inference and not
unstructured latent transfer. The publishable direction is a rate-capped
private tool-trace communication interface with strong source-destroying
controls, byte accounting, latency, and interpretable failure modes.

## Decision

Promote the branch from smoke to strict-small. Stop treating helper-line
copying as the live claim; the live method is now `trace_no_hint` private
tool-trace packet handoff.

## Next Gate

`source_private_hidden_repair_packet_medium_20260429`:

- expand to `500` frozen hidden-repair examples if runtime permits, otherwise
  `320`
- keep `trace_no_hint` as the primary prompt
- run Qwen3 and Phi-3, with `raw_log_no_trace` on Qwen3 as a destruction row
- add paired bootstrap intervals over example IDs
- report packet validity, bytes, generated tokens, p50/p95 source latency, and
  exact ID hash
- pass only if both primary rows remain at least `15` points above target-only
  and all source-destroying controls remain within `2` points of target-only
