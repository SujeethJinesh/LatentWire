# Source-Private Tool-Trace Paper Skeleton

- date: `2026-04-29`
- status: paper skeleton for scoped positive method
- live branch: explicit source-private tool-trace packet handoff
- scale rung: paper framing after large frozen, held-out, seed-repeat, and reviewer-risk gates

## Working Title

Source-Private Tool-Trace Packets for Rate-Capped Evidence Communication

## One-Sentence Claim

A source agent can communicate the explicit private `REPAIR_DIAG` trace field
to a target-side candidate decoder with compact tool-trace packets, improving
candidate selection above target-only and source-destroying controls on frozen
surfaces at `1.55-2.00` bytes versus `366-374` bytes for full hidden-log relay.

## Abstract Skeleton

Agent systems often need one model or agent to use private evidence
observed by another without relaying a full trace. We study this as
source-private communication with decoder side information: the target sees the
public task and a candidate pool, while the source observes a private execution
trace and sends a rate-capped packet. We instantiate the setting with a
hidden-repair benchmark where the target must choose among public repair
candidates and the source sees private tool-trace diagnostics. A compact
explicit `REPAIR_DIAG` packet improves target candidate selection from `25%` to
`80.8-100.0%` across four frozen `500`-example core and held-out-family
surfaces, while zero-source, shuffled-source, random same-byte, answer-only,
answer-masked, target-derived, same-byte structured relay, and trace-removed
controls stay at target-only. The packet averages `1.55-2.00` bytes from model
emitters, compared with roughly `366-374` bytes and `34` tokens for full hidden
log relay. The result is a narrow but reproducible positive method: explicit
source-private tool-trace packets can carry useful hidden source information to
a target-side decoder under strict controls and rate accounting.

## Problem Setup

Let:

- `X`: public task prompt and public candidate pool
- `T`: target-side priors, candidate scores, and candidate metadata
- `S`: source-private evidence, here a hidden execution/tool trace
- `M`: rate-capped source message
- `D`: target-side decoder that selects a candidate using `(X, T, M)`

The central question is whether `M` carries useful source-private information
that cannot be explained by target priors, answer leakage, source-independent
formatting, or same-byte text relay.

Promotion criterion:

- matched source packet beats target/no-source baselines
- source-destroying controls stay near target-only
- same-byte structured relay does not explain compact-budget gains
- exact IDs, bytes, token counts, validity, and candidate-pool recall are
  reported

## Method

### Source Packet

The source receives a private hidden-test/tool log containing an explicit
private trace diagnostic:

```text
private_tool_trace: REPAIR_DIAG=<two-character code>
```

In the model-mediated setting, the source model receives the private log in
`trace_no_hint` mode. The copied helper line and instruction hint are removed;
the source must emit the compact diagnostic packet from the private trace.

The transport packet is the emitted diagnostic code, typically `1.55-2.00`
bytes after parsing model outputs.

### Target Decoder

The target receives the public repair issue, buggy implementation, and a
candidate pool. Each candidate has public candidate metadata including the
diagnostic it handles. The target-side decoder maps a parsed packet to the
candidate whose metadata handles that diagnostic. Invalid packets fall back to
the target prior.

This is intentionally a protocol decoder, not a learned neural target bridge.
The claim is about source-private communication under strict controls, not
latent-state stitching.

## Benchmark

The benchmark contains deterministic Python repair tasks with hidden execution
evidence:

- public issue text
- buggy implementation
- four public candidate repairs
- private hidden-test/tool log visible only to the source
- one gold candidate in every public candidate pool

Surfaces:

- strict-small `160` examples
- medium core `500` examples
- held-out repair-family `500` examples
- seed repeat across core seeds `29/31` and held-out seeds `30/32`

The deterministic control surfaces have candidate-pool recall `1.000`, so the
main reported accuracy is selector accuracy given a fixed public candidate
pool.

## Baselines And Controls

No-source baselines:

- target-only
- target-wrapper/no-source

Source-destroying controls:

- zero-source
- shuffled-source
- random same-byte packet
- target-derived sidecar

Leakage controls:

- answer-only sidecar
- answer-masked sidecar
- helper-template/no-log
- diagnostic-masked full log

Text relay baselines:

- matched-byte truncated hidden-log text
- matched-byte structured JSON relay
- matched-byte concise free-text relay
- full hidden-log relay
- full diagnostic text

Trace-component ablations:

- expected/actual-masked full log
- test-name-masked full log
- trace-removed raw-log prompt

## Main Results

Across four frozen `500`-example surfaces:

| Source model | Surfaces | Matched range | Target | Best controls | Valid packets | Mean bytes |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | `4` | `0.808-0.924` | `0.250` | `0.252-0.258` | `0.776-0.864` | `1.55-1.73` |
| Phi-3-mini | `4` | `1.000` | `0.250` | `0.252-0.258` | `1.000` | `2.00` |

Minimum paired-bootstrap lower bound over target-only:

- Qwen3: `0.516`
- Phi-3: `0.710`

Trace-removal row:

- Qwen3 `raw_log_no_trace` returns to `0.250` with `0` valid packets on all
  four `500`-example surfaces.

## Reviewer-Risk Results

At the `2`-byte paper packet budget:

| Surface | Matched packet | Target | Best source-destroying control | Best reviewer negative | Min reviewer oracle |
|---|---:|---:|---:|---:|---:|
| core seed 29 | `1.000` | `0.250` | `0.254` | `0.250` | `1.000` |
| held-out seed 30 | `1.000` | `0.250` | `0.254` | `0.250` | `1.000` |

Matched-byte JSON/free-text relays, helper-template/no-log, and
diagnostic-masked full logs stay at target-only. Expected/actual-masked and
test-name-masked full logs remain oracles, showing the diagnostic trace field
is the source information actually used.

At `32` bytes, structured JSON and free-text relays become oracles. This is an
expected rate tradeoff and should be shown as a rate curve, not hidden as a
negative.

## Systems Table

| Interface | Mean bytes | Mean tokens | Role |
|---|---:|---:|---|
| Qwen3 model packet | `1.55-1.73` | about `1` | model-produced packet |
| Phi-3 model packet | `2.00` | about `1` | model-produced packet |
| deterministic packet | `2.00` | `1` | oracle protocol |
| full diagnostic text | `14` | `1` | oracle text field |
| full hidden-log relay | `366-374` | about `34` | oracle full trace |

## Threat Model

The claim should be rejected if:

- target-only or no-source wrappers solve the task
- shuffled or random same-byte packets reproduce the gain
- answer-only or answer-masked sidecars reproduce the gain
- target-derived metadata reproduces the gain
- matched-byte structured relay explains compact-budget gains
- the source model succeeds without the private trace diagnostic
- exact ID parity or candidate-pool recall is broken

The current evidence addresses these threats for the scoped protocol claim.

## Interpretability

The packet is directly interpretable: it is a two-character diagnostic code
that maps to candidate metadata. The component-masking rows show which private
trace component matters. Masking expected/actual values or test names preserves
oracle behavior, while masking the diagnostic destroys it. This makes the
communication channel auditable rather than opaque.

## Limitations

- The target-side decoder is deterministic and protocol-shaped.
- Candidate metadata exposes the diagnostic field, so this is a candidate
  selection benchmark rather than unconstrained code generation.
- The benchmark is synthetic but includes held-out repair families and seed
  repeats.
- The method is explicit tool-trace packet communication, not learned latent
  transfer.
- A learned or LLM-mediated target decoder would strengthen the paper but is not
  part of the current claim.
- Structured text relay becomes an oracle when the byte budget is large enough
  to carry the diagnostic.

## Paper Figures And Tables

Required:

- Figure 1: source-private communication setup `(X, T, S, M, D)`
- Table 1: model-mediated packet rows over four frozen surfaces
- Table 2: deterministic controls and reviewer-risk rows at 2 bytes
- Figure 2: rate curve: compact packet, JSON/free-text relay, full diagnostic
  text, full hidden-log relay
- Table 3: systems bytes/tokens and validity
- Table 4: threat-model controls and pass/fail summary

## Submission Strategy

Main submission claim:

> A reproducible source-private communication protocol where compact
> model-produced tool-trace packets transfer hidden source evidence to a
> target-side candidate decoder under strict source-destroying controls.

Avoid framing as:

- “latent wire” in the sense of hidden-state transfer
- general cross-LLM internal-state communication
- raw-log repair reasoning

Reviewer positioning:

- Strong positive method if the paper sells rate-capped private evidence
  communication as the contribution.
- Weak or overclaimed if framed as learned latent bridging.

## Next Gate

`source_private_tool_trace_target_decoder_smoke_20260429`:

- replace the deterministic `REPAIR_DIAG -> candidate metadata` lookup with an
  LLM-mediated or learned target-side selector on a small frozen slice
- require it to recover a large fraction of the deterministic gain while
  preserving zero/shuffled/random and same-byte relay controls
- if it fails, keep the deterministic protocol decoder as the scoped claim and
  list learned target decoders as a limitation
