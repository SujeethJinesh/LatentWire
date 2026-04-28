# Source-Private Hidden-Repair Weakened-Helper Gate

- date: `2026-04-28`
- status: promoted weakened-helper smoke, not yet ICLR-ready
- live branch: source-private hidden-repair packet handoff
- scale rung: weakened-helper smoke

## Question

Does the hidden-repair packet result survive after removing the most direct
prompt scaffolding, and does it fail when the source-private diagnostic trace is
destroyed?

## Prompt Modes

- `copied_helper`: full private log plus a copied `REPAIR_DIAG` helper line.
- `log_only`: full private log only; no copied helper line.
- `trace_no_hint`: private log with `REPAIR_DIAG` trace line but no copied
  helper line and no hint.
- `raw_log_no_trace`: private execution log with both the `REPAIR_DIAG` trace
  and hint removed.

## Results

| Run | Model | Prompt mode | Pass | Matched | Target-only | Best control | Valid packets | Mean bytes | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_log_only | Qwen/Qwen3-0.6B | log_only | `true` | 0.984 | 0.250 | 0.250 | 0.984 | 1.97 | 366.37 |
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.781 | 0.250 | 0.250 | 0.734 | 1.47 | 293.33 |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.250 | 0.000 | 0.00 | 277.67 |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 2.00 | 426.51 |

## Interpretation

The copied helper line is not necessary. Qwen3 remains strong in `log_only`,
and both Qwen3 and Phi-3 pass in `trace_no_hint`, where the source sees the
private trace line but no copied helper line or hint. Removing the trace itself
destroys the signal: Qwen3 emits no valid repair packets and returns to
target-only accuracy.

This strengthens the source-private communication story, but it does not yet
solve the main reviewer objection. The benchmark still uses an explicit
source-private diagnostic field and candidate-side diagnostic metadata. The
defensible claim is now narrower and cleaner: models can transmit a compact
private tool-trace field across model families, and the target can use it under
strict source-destroying controls.

## Decision

Promote the branch to weakened-helper smoke. The trace field is the live
communication interface; the copied helper line and hint can be removed.

## Next Gate

`source_private_hidden_repair_packet_strict_small_20260429`:

- scale the hidden-repair benchmark from `64` to `160` frozen examples
- use `trace_no_hint` as the primary source prompt
- include Qwen3 and Phi-3 source emitters
- retain `raw_log_no_trace` as a source-signal destruction control
- pass only if matched remains at least `15` points above target-only, all
  source-destroying controls remain flat, exact ID parity holds, and packet
  validity/bytes/latency are reported
