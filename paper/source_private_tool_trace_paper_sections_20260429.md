# Source-Private Tool-Trace Paper Sections

- date: `2026-04-29`
- status: first section-level draft for scoped positive method
- live branch: explicit source-private tool-trace packet handoff
- scale rung: paper-section drafting

## Title

Source-Private Tool-Trace Packets for Rate-Capped Evidence Communication

## Abstract

Agent systems often need one model to use private evidence observed by another
without relaying a full trace. We formalize this as source-private evidence
communication with decoder side information: a target agent receives the public
task and a candidate pool, while a source agent observes private execution
evidence and sends a rate-capped message. We instantiate this setting with a
hidden-repair benchmark where the target selects among public repair candidates
and the source observes private tool-trace diagnostics. A compact explicit
`REPAIR_DIAG` packet improves target-side candidate selection from `25%` to
`80.8-100.0%` across four frozen `500`-example core and held-out-family
surfaces. Zero-source, shuffled-source, random same-byte, answer-only,
answer-masked, target-derived, same-byte structured relay, helper/no-log, and
trace-removed controls remain at target-only. Model-produced packets average
`1.55-2.00` bytes, compared with roughly `366-374` bytes and `34` tokens for
full hidden-log relay. A small Qwen3 target-decoder smoke further shows that the
packet can be consumed by a model-mediated target selector, not only by a
hand-coded lookup. The result is a narrow but reproducible positive method:
explicit source-private tool-trace packets can transmit useful hidden source
evidence under strict controls and rate accounting.

## Introduction Draft

Cross-model and multi-agent systems increasingly split work across components
with asymmetric information. A tool-using source agent may observe execution
traces, retrieval results, or private memory unavailable to a target agent that
must make the final decision. The naive solution is to relay the entire trace,
but full trace relay is expensive, slow, and often unnecessary when the target
already has strong side information such as a candidate pool or cached prior.

We study the narrower question: can a source agent send a compact, auditable
message that changes the target's decision only when matched private source
evidence is present? This framing turns cross-agent communication into a
source-private coding problem. The target has public task state `X` and
target-side candidate state `T`; the source has private evidence `S`; the
source sends a rate-capped message `M`; and the target decoder selects a final
candidate using `(X, T, M)`.

Our main contribution is a reproducible positive method and benchmark for this
setting. In hidden-repair candidate selection, the target sees the public issue,
buggy implementation, and repair candidates. The source sees a private
tool-trace diagnostic. A source model emits a compact `REPAIR_DIAG` packet, and
the target-side decoder uses it to select the candidate whose public metadata
matches the private diagnostic. The method is intentionally explicit and
interpretable: the packet is a two-character diagnostic, not an opaque hidden
state.

The result should be read with the right boundary. We do not claim unstructured
latent transfer, raw-log repair reasoning, or universal cross-model
communication. We claim that a rate-capped source-private packet can carry
useful hidden evidence to a target-side candidate decoder, under controls that
destroy source information, answer leakage, target priors, and same-byte
structured relays.

## Problem Formulation

Each example consists of:

- public task and candidate pool `X`
- target-side prior/candidate state `T`
- source-private trace evidence `S`
- rate-capped source message `M`
- target decoder `D(X, T, M)`

The evaluation asks whether `D(X, T, M_matched)` improves over `D(X, T, empty)`
and over source-destroying controls. A method is not considered a communication
success unless the gain disappears under zero-source, shuffled-source,
random-same-byte, answer-only, answer-masked, target-derived, and same-byte
relay controls.

We report:

- selector accuracy
- candidate-pool recall
- exact ID parity
- packet validity
- paired uncertainty
- bytes and token counts
- source/control deltas

## Method

The source prompt contains a private hidden-test log with an explicit diagnostic
line:

```text
private_tool_trace: REPAIR_DIAG=<code>
```

In the main model-mediated setting, the source uses `trace_no_hint` mode: the
copied helper line and hint are removed, but the private diagnostic line remains
inside the private log. The source model must output only the compact packet.
Invalid packets are treated as no-source fallbacks.

The target-side protocol decoder receives the public candidate pool and the
packet. It chooses the candidate whose `handles_repair_diag` metadata matches
the parsed packet; if no packet is valid or no candidate matches, it falls back
to the target prior. This decoder is deliberately simple so that the experiment
isolates whether the source message carries useful private evidence.

We also include a Qwen3 target-decoder smoke: the same source packets are fed to
a target LLM with candidate labels, candidate diagnostic metadata, and a
target-prior fallback. This ablation tests whether the hand-coded lookup can be
replaced by a model-mediated selector on small slices.

## Benchmark

The hidden-repair benchmark contains deterministic Python repair tasks. Each
example has:

- public issue text
- buggy implementation
- four public candidate repairs
- one gold candidate in the candidate pool
- source-private hidden-test/tool log
- a private `REPAIR_DIAG` code identifying the hidden repair diagnostic

The main surfaces are:

- strict-small `160` examples
- medium core `500` examples
- held-out repair-family `500` examples
- seed-repeat surfaces: core seeds `29/31`, held-out seeds `30/32`

The held-out split uses disjoint repair families, and deterministic surfaces
have candidate-pool recall `1.000`, separating candidate generation from
candidate selection.

## Results

Across four frozen `500`-example surfaces, model-produced packets improve
target-side selection substantially:

| Source model | Surfaces | Matched range | Target | Best controls | Valid packets | Mean bytes |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | `4` | `0.808-0.924` | `0.250` | `0.252-0.258` | `0.776-0.864` | `1.55-1.73` |
| Phi-3-mini | `4` | `1.000` | `0.250` | `0.252-0.258` | `1.000` | `2.00` |

The minimum paired-bootstrap lower bound over target-only is `0.516` for Qwen3
and `0.710` for Phi-3. Removing the private trace diagnostic returns Qwen3 to
target-only (`0.250`) with `0` valid packets on all four surfaces.

## Controls And Threat Model

The main controls address the following reviewer threats:

| Threat | Control | Result |
|---|---|---|
| target prior solves task | target-only, wrapper/no-source | `0.250` |
| packet works without matched source | zero, shuffled, random same-byte | `0.252-0.258` on large surfaces |
| answer leakage explains gain | answer-only, answer-masked | target-only |
| target metadata explains gain | target-derived sidecar | target-only |
| same-byte text explains gain | truncated hidden-log, JSON, free text | target-only at `2` bytes |
| raw log is enough | `raw_log_no_trace` | target-only with `0` valid packets |
| trace component artifact | diagnostic-masked full log | target-only |

Expected/actual-masked and test-name-masked full logs remain oracles, showing
that the private diagnostic field is the useful source information.

## Rate And Systems Analysis

The systems value is rate efficiency. Model-produced packets average
`1.55-2.00` bytes and roughly one token. Full hidden-log relay reaches oracle
accuracy but costs roughly `366-374` bytes and `34` tokens per example.

Structured JSON and free-text relays fail at the compact `2`-byte budget, but
become oracles at `32` bytes because the diagnostic is then exposed in a
parseable form. The paper should present this as a rate curve, not as a hidden
failure: text relay is a valid baseline when granted enough bytes, while the
packet method occupies the far-left, compact-rate regime.

## Target-Decoder Smoke

To reduce the concern that the method is only a hard-coded lookup, we ran a
Qwen3 target-side selector that receives candidate labels, candidate diagnostic
metadata, a target-prior fallback, and the source packet.

| Surface | N | Target | Matched packet | Best control | Pass |
|---|---:|---:|---:|---:|---|
| core seed 29 | `16` | `0.250` | `0.688` | `0.250` | `True` |
| held-out seed 30 | `32` | `0.250` | `0.750` | `0.281` | `True` |

This row is a smoke ablation, not the main evidence. It shows that the
candidate-selection step can be model-mediated while preserving source
controls on small slices.

## Interpretability

The method is interpretable by construction. The packet is the diagnostic code,
the target candidate metadata exposes which candidate handles each diagnostic,
and component masking identifies which private trace field carries the signal.
This makes successes and failures auditable at the example level.

## Limitations

- The main target decoder is deterministic and protocol-shaped.
- Candidate metadata exposes the diagnostic mapping.
- The benchmark is synthetic, though held-out families and seed repeats reduce
  template-specific concerns.
- The result is source-private evidence communication, not learned latent
  transfer.
- The target-decoder smoke is small and should not be overclaimed.
- Structured text relay becomes an oracle when granted enough bytes.

## Conclusion Draft

The evidence supports a narrow positive method for cross-agent communication:
compact explicit source-private tool-trace packets can transmit hidden source
evidence to a target-side candidate decoder under strict controls. The result is
not a universal latent bridge, but it is reproducible, interpretable,
rate-accounted, and robust across large frozen slices, held-out repair families,
seed repeats, reviewer-risk controls, and a small model-mediated target-decoder
ablation.

## Next Gate

`source_private_tool_trace_paper_draft_20260430`:

- convert these sections into the paper source or a full markdown draft
- add figure/table placeholders for main table, threat model, and rate curve
- decide whether to scale the target-decoder smoke or leave it as an ablation
