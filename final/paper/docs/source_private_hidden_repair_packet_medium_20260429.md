# Source-Private Hidden-Repair Packet Medium Gate

- date: `2026-04-29`
- status: promoted medium confirmation, not yet ICLR-ready
- live branch: explicit source-private tool-trace packet handoff
- scale rung: medium confirmation

## Question

Does the strict-small hidden-repair packet result survive on a larger frozen
slice with paired uncertainty?

## Setup

The medium gate uses `500` frozen hidden-repair examples from the same repair
families. The primary source prompt is `trace_no_hint`: the copied helper line
and hint are removed, but the private hidden execution log still contains the
explicit `REPAIR_DIAG` tool-trace field. Qwen3 and Phi-3 are primary source
emitters. Qwen3 `raw_log_no_trace` is retained as the aligned source-signal
destruction row.

## Results

Deterministic packet sweep:

| Budget bytes | Matched | Best no-source | Best control | Matched text | Full log |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 0.250 | 0.254 | 0.250 | 1.000 |
| 4 | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 8 | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 |
| 16 | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 |
| 32 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |

Model-produced packets with paired bootstrap intervals:

| Run | Model | Mode | Matched | Target | Best control | Valid | Mean bytes | p95 latency ms | Delta target 95% CI | Delta control 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.252 | 0.776 | 1.55 | 522.87 | [0.516, 0.600] | [0.514, 0.602] |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | 2.00 | 516.41 | [0.714, 0.788] | [0.708, 0.786] |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | 0.00 | 365.93 | [0.000, 0.000] | [-0.006, 0.000] |

## Interpretation

This is now a real positive-method candidate rather than only a smoke result.
The method has:

- a clear source-private signal: `REPAIR_DIAG` in a hidden tool trace
- a compact transmitted message: roughly `1.55-2.00` bytes on successful rows
- strong target/control separation on `500` frozen examples
- source destruction: removing the trace returns to target-only
- cross-model support: Qwen3 and Phi-3 both pass
- paired uncertainty: primary lower bounds remain far above the `+0.15` gate

The method is still explicitly scoped. It is not a claim that arbitrary raw logs
are enough; the trace field is the communication interface. That scope is now
paper-defensible if framed as private tool-trace communication rather than
latent transfer.

## Decision

Promote to medium confirmation. The next work should stop tuning the protocol
and start strengthening paper readiness: held-out families, seed repeats, and a
competitor baseline section.

## Next Gate

`source_private_hidden_repair_packet_holdout_families_20260429`:

- add held-out repair families not used by the current eight templates
- freeze `200-500` examples
- keep the same `trace_no_hint` protocol and controls
- require Qwen3 and Phi-3 to remain above target-only by at least `15` points
- keep `raw_log_no_trace`, shuffled, random same-byte, answer-only,
  answer-masked, and target-derived controls flat
