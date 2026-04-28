# Source-Private Tool-Trace Paper Claim Draft

- date: `2026-04-29`
- status: scoped paper-claim draft, not final submission text
- live branch: explicit source-private tool-trace packet handoff
- scale rung: large frozen slice with held-out families and seed repeats

## Candidate Paper Claim

We study a source-private communication setting where a source agent observes
private execution/tool traces and sends a rate-capped packet to a target-side
candidate decoder. A compact explicit trace packet, `REPAIR_DIAG`, improves
repair-candidate selection from `25%` target-only accuracy to `80.8-100.0%`
across four frozen `500`-example core and held-out-family surfaces, while
zero-source, shuffled-source, random same-byte, answer-only, answer-masked,
target-derived, matched-byte hidden-log text, and trace-removed controls stay
at target-only.

This is a positive cross-agent communication result for explicit private
tool-trace packets. It is not a claim that raw logs alone are sufficient, not a
claim of unstructured latent transfer, and not yet a claim about a learned
target-side neural decoder.

## Paper Story

The target receives the public repair prompt, a candidate pool, and its own
candidate metadata. The source receives private execution evidence that is not
available to the target. The source emits a rate-capped packet containing the
private repair diagnostic. The target-side decoder uses the packet to select
the candidate whose metadata matches the private diagnostic.

The core contribution is the communication protocol and benchmark: a
source-private evidence packet can carry useful information that target priors,
answer labels, shuffled packets, random same-byte packets, and matched-byte
truncated text cannot reproduce.

## Method Boundary

Claim:

- explicit source-private tool-trace packet handoff
- compact diagnostic packets, usually `1.55-2.00` bytes from model emitters
- deterministic target-side protocol decoder for candidate selection
- source model emits packets from private traces in `trace_no_hint` mode
- target-only and source-destroying controls remain near `25%`

Do not claim:

- raw-log repair inference
- learned latent transfer between LLM internals
- universal cross-model communication
- a learned target-family bridge
- replacement of full code-repair systems

## Evidence Table

| Surface | Source model | Matched | Target | Best control | Valid packets | Delta target 95% CI |
|---|---|---:|---:|---:|---:|---:|
| core seed 29 | Qwen3-0.6B | `0.808` | `0.250` | `0.252` | `0.776` | `[0.516, 0.600]` |
| core seed 29 | Phi-3-mini | `1.000` | `0.250` | `0.252` | `1.000` | `[0.714, 0.788]` |
| core seed 31 | Qwen3-0.6B | `0.808` | `0.250` | `0.256` | `0.776` | `[0.516, 0.602]` |
| core seed 31 | Phi-3-mini | `1.000` | `0.250` | `0.256` | `1.000` | `[0.710, 0.786]` |
| holdout seed 30 | Qwen3-0.6B | `0.922` | `0.250` | `0.258` | `0.864` | `[0.632, 0.712]` |
| holdout seed 30 | Phi-3-mini | `1.000` | `0.250` | `0.258` | `1.000` | `[0.710, 0.788]` |
| holdout seed 32 | Qwen3-0.6B | `0.924` | `0.250` | `0.252` | `0.860` | `[0.634, 0.716]` |
| holdout seed 32 | Phi-3-mini | `1.000` | `0.250` | `0.252` | `1.000` | `[0.710, 0.786]` |

The trace-removed `raw_log_no_trace` rows return Qwen3 to `0.250` matched
accuracy with `0` valid packets on all four surfaces.

## Systems Claim

The packet interface has a clear systems advantage over full private log relay
for this benchmark. Model-produced packets average `1.55-2.00` bytes. The full
hidden-log relay reaches oracle accuracy, but costs roughly `366-374` bytes and
`34` tokens per example on the representative `500`-example surfaces. A
matched-byte hidden-log text baseline remains at `0.250`, so the advantage is
not explained by giving the target the first few bytes of the private log.

## Baselines And Controls

Already covered:

- target-only and target-wrapper/no-source
- zero-source
- shuffled-source
- random same-byte packet
- answer-only sidecar
- answer-masked sidecar
- target-derived sidecar
- matched-byte truncated hidden-log text
- full hidden-log relay oracle
- full diagnostic oracle
- trace-removed raw-log prompt
- held-out repair families
- seed repeats
- two source emitter families

Remaining reviewer-risk rows:

- matched-byte structured JSON and concise free-text relay
- target helper-only/no-log oracle
- trace-component masking: expected/actual value, line number, and test name
- paper table separation of candidate-pool recall and selector accuracy
- optional learned or LLM-mediated target-family row

## Reviewer-Safe Abstract Paragraph

We introduce a source-private communication benchmark for cross-agent repair
selection. A source agent observes private execution traces and must transmit a
rate-capped packet to a target-side candidate decoder. In a `500`-example
hidden-repair benchmark with core and held-out repair families, compact
model-produced tool-trace packets improve target candidate selection from
`25%` to `80.8-100.0%`, while zero-source, shuffled-source, random same-byte,
answer-only, answer-masked, target-derived, matched-byte text, and
trace-removed controls remain at target-only. The result supports a narrow but
reproducible positive method: explicit private tool-trace packets can carry
useful source information at far lower byte/token cost than full private log
relay.

## Next Gate

`source_private_tool_trace_reviewer_risk_rows_20260429`:

- add structured matched-byte JSON/free-text relays
- add helper-only/no-log target oracle
- add trace-component masking ablations
- report candidate-pool recall separately from selector accuracy
- rerun the cheapest decisive slice first, then widen only if the exact claim
  changes
