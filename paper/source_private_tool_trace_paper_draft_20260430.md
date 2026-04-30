# Source-Private Tool-Trace Packets for Rate-Capped Evidence Communication

Status: full markdown paper draft, converted from the evidence memos on
`2026-04-30`.

## Abstract

Agent systems can benefit when one model uses private evidence observed by another
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
explicit source-private tool-trace packets can transmit the benchmark's hidden
diagnostic evidence under the controls tested here and with explicit rate
accounting.

## 1. Introduction

Multi-agent and cross-model systems increasingly divide work across components
with asymmetric information. A tool-using source agent may observe execution
traces, retrieval results, private memory, or environment feedback unavailable
to the target agent that must make the final decision. The simplest interface
is full text relay: send the whole trace and let the target reason over it. But
full relay is often wasteful. If the target already has a strong prior,
candidate pool, or cached state, the source may only need to communicate the
residual private evidence that resolves the target's uncertainty.

This paper studies a deliberately narrow version of that problem. The target
sees a public task and candidate pool. The source sees private execution
evidence. The source sends a compact message. The target combines its public
side information with the message to choose a candidate. We ask: can a model
emit a rate-capped, interpretable source-private packet that improves the
target only when matched private evidence is present?

The framing follows the intuition of distributed source coding and
rate-distortion theory: a sender can communicate fewer bits when the decoder
has useful side information, and the correct objective is task distortion at a
communication rate rather than reconstruction of the entire source state.
Here, the target's candidate pool is decoder side information, and the source
packet is evaluated by downstream selector accuracy under strict source
controls.

We instantiate this idea in a hidden-repair candidate-selection benchmark. The
target receives a public repair issue, buggy implementation, and four public
repair candidates. The source receives a private hidden-test/tool trace
containing an explicit repair diagnostic. The source model emits a compact
`REPAIR_DIAG` packet; the target-side decoder uses the packet and public
candidate metadata to select a repair candidate. The packet is interpretable,
auditable, and byte-counted.

Our results support a scoped positive method. Across four frozen `500`-example
surfaces, Qwen3-0.6B source packets reach `80.8-92.4%` selector accuracy and
Phi-3-mini packets reach `100%`, while target-only accuracy is `25%` and
source-destroying controls remain near `25%`. On these frozen synthetic
surfaces, the result is unchanged under held-out repair families, seed repeats,
answer-leakage controls, target-derived sidecars, trace removal, and same-byte
structured text controls. A full hidden log relay is also an oracle, but costs
roughly two orders of magnitude more bytes.

The contribution is not a learned latent bridge or a universal cross-model
communication interface. It is a reproducible, interpretable, rate-capped
source-private evidence packet that survives a reviewer-style threat model.
That boundary is central: the method is explicit tool-trace packet
communication for candidate selection with decoder side information.

**Contributions.**

- We define a source-private evidence-communication setting with public target
  side information, private source evidence, rate-capped messages, and
  source-destroying controls.
- We introduce a hidden-repair candidate-selection benchmark with frozen core
  and held-out-family surfaces, exact ID parity, candidate-pool recall
  accounting, byte/token telemetry, and reviewer-risk controls.
- We show that compact model-produced `REPAIR_DIAG` packets improve target
  selection from `25%` to `80.8-100.0%` across four `500`-example surfaces.
- We demonstrate rate efficiency relative to full hidden-log relay and
  same-byte structured relay controls.
- We include a small Qwen3 target-decoder smoke showing that packet consumption
  can be model-mediated rather than purely hard-coded.

## 2. Problem Setup

Each example has:

- `X`: public task information and public candidate pool;
- `T`: target-side priors, candidate scores, and candidate metadata;
- `S`: source-private evidence, here a private hidden-test/tool trace;
- `M`: source message under a byte budget;
- `D`: target decoder selecting a final candidate from `(X, T, M)`.

The target never sees `S` directly unless a text-relay oracle baseline is being
evaluated. The source does not choose a candidate label directly in the primary
method; it emits a compact packet. The target decoder either uses the packet or
falls back to the target prior.

The central metric is selector accuracy:

```text
Acc(D(X, T, M_matched))
```

relative to no-source and source-destroying controls:

```text
D(X, T, empty), D(X, T, M_zero), D(X, T, M_shuffled),
D(X, T, M_random), D(X, T, M_answer), D(X, T, M_target)
```

A method is only a communication success if the matched source packet improves
accuracy and the improvement disappears when source information is destroyed or
replaced by target-derived/answer-derived sidecars. We also require exact ID
parity, candidate-pool recall, byte/token accounting, and paired uncertainty.

## 3. Benchmark: Hidden-Repair Candidate Selection

The benchmark consists of deterministic Python repair tasks. Each example
contains a public issue, buggy implementation, four public repair candidates,
and a private hidden-test/tool log visible only to the source. Each public
candidate has metadata:

```text
handles_repair_diag=<code>
```

The private source log contains:

```text
private_tool_trace: REPAIR_DIAG=<code>
```

The gold candidate is the candidate whose public metadata handles the private
diagnostic. The target prior deliberately solves one quarter of the examples:
target-only accuracy is `25%`. The remaining cases require the private source
packet to select the correct candidate.

**Surfaces.**

- Strict-small: `160` frozen examples.
- Medium core: `500` frozen examples.
- Held-out families: `500` frozen examples from disjoint repair families.
- Seed repeats: core seeds `29/31`; held-out seeds `30/32`.

The deterministic surfaces have candidate-pool recall `1.000`, so selector
accuracy is not confounded with candidate generation recall.

**Figure 1.** Source-private evidence communication setup:
`X,T` at target, `S` at source, rate-capped packet `M`, decoder `D`.
Asset: `results/source_private_tool_trace_latex_or_figures_20260430/source_private_setup.svg`.

## 4. Method: Tool-Trace Packets

The source model receives a private hidden-test log and emits only a compact
repair packet. In the main model-mediated condition, `trace_no_hint`, the
private diagnostic line remains in the private log, but the copied helper line
and hint are removed. The source prompt asks the model to output only the
two-character packet. Parsed invalid packets fall back to the target prior.

The target-side protocol decoder receives the public candidate pool and the
packet. If the packet is a valid diagnostic matching exactly one candidate's
`handles_repair_diag` metadata, the decoder selects that candidate. Otherwise
it returns the target-prior candidate.

This design is intentionally explicit. It makes the communication channel
auditable, keeps the byte budget clear, and enables strong source-destroying
controls. The tradeoff is that the target decoder is protocol-shaped rather
than a learned latent bridge. We therefore include a Qwen3 target-decoder smoke
as an ablation, but keep the main claim scoped to the protocol decoder.

## 5. Baselines And Controls

We compare against:

**No-source baselines.**

- target-only prior;
- target wrapper/no-source.

**Source-destroying controls.**

- zero-source packet;
- shuffled-source packet from a different example;
- random same-byte packet;
- target-derived sidecar.

**Leakage controls.**

- answer-only sidecar;
- answer-masked sidecar;
- helper-template/no-log;
- diagnostic-masked full log.

**Text relay baselines.**

- matched-byte truncated hidden-log text;
- matched-byte structured JSON relay;
- matched-byte concise free-text relay;
- full hidden-log relay;
- full diagnostic text.

**Trace-component ablations.**

- expected/actual-masked full log;
- test-name-masked full log;
- raw log with trace removed.

## 6. Main Results

Table 1 is the primary paper table. It reports model-produced source packets
across four frozen `500`-example surfaces.

**Table 1. Model-mediated source-packet rows.**

| Surface | Source model | Mode | Matched | Target | Best control | Valid | Bytes | Delta target 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|
| core seed 29 | Qwen3-0.6B | `trace_no_hint` | `0.808` | `0.250` | `0.252` | `0.776` | `1.55` | `[0.516, 0.600]` |
| core seed 29 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.252` | `1.000` | `2.00` | `[0.714, 0.788]` |
| core seed 31 | Qwen3-0.6B | `trace_no_hint` | `0.808` | `0.250` | `0.256` | `0.776` | `1.55` | `[0.516, 0.602]` |
| core seed 31 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.256` | `1.000` | `2.00` | `[0.710, 0.786]` |
| held-out seed 30 | Qwen3-0.6B | `trace_no_hint` | `0.922` | `0.250` | `0.258` | `0.864` | `1.73` | `[0.632, 0.712]` |
| held-out seed 30 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.258` | `1.000` | `2.00` | `[0.710, 0.788]` |
| held-out seed 32 | Qwen3-0.6B | `trace_no_hint` | `0.924` | `0.250` | `0.252` | `0.860` | `1.72` | `[0.634, 0.716]` |
| held-out seed 32 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.252` | `1.000` | `2.00` | `[0.710, 0.786]` |

The lower-confidence Qwen3 rows are driven by packet validity: invalid packets
fall back to the target prior. Phi-3 emits valid packets on all examples. The
minimum paired-bootstrap lower bound over target-only is `0.516` for Qwen3 and
`0.710` for Phi-3.

Removing the private trace diagnostic kills the effect. Qwen3 `raw_log_no_trace`
returns to `0.250` with `0` valid packets on all four `500`-example surfaces.

## 7. Threat-Model Results

Table 2 summarizes deterministic controls on representative core and held-out
surfaces at the `2`-byte paper packet budget.

**Table 2. Deterministic controls at 2 bytes.**

| Surface | Matched | Target | Wrapper | Zero | Shuffled | Random | Answer-only | Answer-masked | Target-derived | Full log | Full diag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core seed 29 | `1.000` | `0.250` | `0.250` | `0.250` | `0.250` | `0.254` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |
| held-out seed 30 | `1.000` | `0.250` | `0.250` | `0.250` | `0.250` | `0.254` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |

Table 3 addresses reviewer-risk rows for structured relay and leakage.

**Table 3. Reviewer-risk controls at 2 bytes.**

| Surface | Hidden-log text | JSON relay | Free-text relay | Helper/no-log | Diag-masked log | Exp/actual masked | Test-name masked |
|---|---:|---:|---:|---:|---:|---:|---:|
| core seed 29 | `0.250` | `0.250` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |
| held-out seed 30 | `0.250` | `0.250` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |

The diagnostic-masked full log collapses to target-only, while expected/actual
and test-name masked logs remain oracles. This identifies the private
diagnostic field as the information-carrying trace component.

## 8. Rate And Systems Analysis

The method's systems value is rate efficiency. Full hidden-log relay is a valid
oracle baseline, but it is much larger than the packet. Table 4 summarizes the
transport costs.

**Table 4. Systems and rate comparison.**

| Interface | Mean bytes | Mean tokens | Role |
|---|---:|---:|---|
| Qwen3 model packet | `1.55-1.73` | about `1` | model-produced packet |
| Phi-3 model packet | `2.00` | about `1` | model-produced packet |
| deterministic packet | `2.00` | `1` | oracle protocol |
| full diagnostic text | `14` | `1` | oracle text field |
| full hidden-log relay | `366-374` | about `34` | oracle full trace |

**Figure 2.** Accuracy-versus-byte curve for compact packet, matched-byte
hidden-log text, structured JSON relay, concise free-text relay, full
diagnostic text, and full hidden-log relay. Asset:
`results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.svg`.
Source data:
`results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.csv`.

At `2` bytes, structured JSON and free-text relays remain target-only. At
`32` bytes, they become oracles because the diagnostic is visible in a parseable
format. This is not a contradiction; it is the expected rate curve. The packet
method occupies the compact-rate regime.

## 9. Target-Decoder Ablation

A skeptical reviewer may object that the target-side protocol decoder is doing
too much work. To test this, we replace the hard-coded packet-to-candidate
lookup with Qwen3-0.6B acting as a target-side selector. The model receives the
target-prior fallback label, the source packet, candidate labels, and candidate
`handles_repair_diag` metadata.

**Table 5. Qwen3 target-decoder ablation.**

| Surface | N | Target | Matched packet | Best control | Pass |
|---|---:|---:|---:|---:|---|
| core seed 29 | `16` | `0.250` | `0.688` | `0.250` | yes |
| held-out seed 30 | `32` | `0.250` | `0.750` | `0.281` | yes |
| core seed 29 | `160` | `0.250` | `0.694` | `0.250` | yes |

This is still an ablation, not the main systems evidence. It reduces the
hand-coded lookup concern by showing that packet consumption can be
model-mediated while preserving controls. The `n=160` row has paired CI95 lower
bounds `+0.369` versus both target-only and best control, but CPU decoding is
slow (`2670 ms` p50 per matched condition), so it should not be used as a
serving-speed claim.

## 10. Interpretability

The packet channel is directly auditable in this benchmark. The packet is a
two-character diagnostic code; target candidate metadata states which
diagnostic each candidate handles; and component-masking controls identify the
private trace field that carries the signal. Failures can be audited as invalid
source packets, fallback-to-prior cases, or target-decoder selection mistakes.

This interpretability is a deliberate design choice. It trades off the ambition
of unstructured latent transfer for a clear, reproducible communication
contract that survives source-destroying controls.

## 11. Related Work

Distributed source coding motivates our problem formulation: when a decoder has
side information, the sender need not transmit the full source state.
Slepian-Wolf coding, Wyner-Ziv coding, and rate-distortion theory provide the
high-level lens of residual information under a rate constraint. Our setting is
task-oriented rather than reconstruction-oriented: the target uses side
information to select a candidate, and success is measured by candidate
accuracy per communicated byte.

Semantic communication and tool-using agents provide the systems motivation.
In tool-using workflows, private observations such as execution logs or
retrieval results are often the decisive evidence. Rather than relay a whole
trace, our method communicates a compact diagnostic packet.

VLM connector and JEPA-style architectures motivate future learned extensions:
query bottlenecks, gated adapters, and anti-collapse objectives could replace
explicit packets once a robust source-private benchmark exists. We deliberately
do not lead with those architectures because previous ordinary math surfaces
were dominated by target-prior contamination. The current benchmark first
establishes a clean source-private communication surface.

Cache/activation transfer methods such as C2C and KV communication are relevant
competitors for broader cross-model communication. They transfer internal state
or caches; our method instead transfers an explicit source-private evidence
packet. These are complementary regimes, and future work should compare them on
shared private-evidence tasks.

## 12. Limitations

The main limitation is scope. The benchmark exposes a public diagnostic field
in candidate metadata and a private diagnostic field in the source trace. This
makes the problem auditable and rate-counted, but it is not unconstrained code
generation or raw-log repair reasoning.

The primary target decoder is deterministic and protocol-shaped. The Qwen3
target-decoder ablation now passes at `n=160` on the core surface, which
reduces this concern, but held-out `n=160` and product-codebook-specific
model-mediated decoding remain open.

The benchmark is synthetic. Held-out repair families and seed repeats reduce
template-specific concerns, but real tool traces, retrieval traces, and
multi-agent workflows are needed before claiming external deployment value.

Structured text relay becomes an oracle when granted enough bytes. The systems
claim is therefore a compact-rate claim: source-private packets are more
efficient at the measured byte budget, not categorically better than all text
communication at all rates.

Finally, this work does not show unstructured latent transfer, internal-state
communication, or universal cross-LLM communication. It shows a concrete,
interpretable positive method for source-private evidence packets.

## 13. Reproducibility

All promoted claims use frozen example IDs, exact ID parity, byte accounting,
and append-only result manifests. Main artifacts:

- `results/source_private_tool_trace_baseline_pack_20260429/`
- `results/source_private_tool_trace_reviewer_risk_rows_20260429/`
- `results/source_private_tool_trace_target_decoder_smoke_20260429/`
- `paper/source_private_tool_trace_final_table_20260429.md`
- `paper/source_private_tool_trace_target_decoder_smoke_20260429.md`

Core scripts:

- `scripts/run_source_private_hidden_repair_packet_smoke.py`
- `scripts/run_source_private_hidden_repair_packet_llm.py`
- `scripts/summarize_source_private_hidden_repair_medium.py`
- `scripts/build_source_private_tool_trace_baseline_pack.py`
- `scripts/run_source_private_tool_trace_target_decoder_smoke.py`

## 14. Conclusion

We presented a scoped positive method for source-private evidence
communication. A source model reads private tool-trace diagnostics and emits a
compact explicit packet. A target-side candidate decoder combines that packet
with public candidate metadata to select the correct repair. The method is
reproducible, interpretable, rate-accounted, and stable across the tested
frozen `500`-example surfaces, held-out repair families, seed repeats,
source-destroying controls, structured relay baselines, and a small target-LLM
decoder ablation. The result
does not solve broad latent transfer, but it establishes a defensible
source-private communication protocol that can be extended toward learned
decoders and less structured private evidence.

## Appendix A. Exact Claim For Submission

We claim that explicit source-private tool-trace packets can communicate hidden
execution evidence to a target-side candidate decoder under rate constraints.
At `1.55-2.00` bytes, model-produced packets improve hidden-repair candidate
selection from `25%` target-only to `80.8-100.0%` across four frozen
`500`-example surfaces, while source-destroying, answer, target-derived,
same-byte structured relay, and trace-removal controls remain at target-only.

We do not claim raw-log repair inference, learned latent transfer, universal
cross-model communication, or unconstrained repair generation.

## Appendix B. Figure/Table Checklist

- Figure 1: source-private setup diagram. Asset generated:
  `results/source_private_tool_trace_latex_or_figures_20260430/source_private_setup.svg`.
- Figure 2: rate curve over packet/text/full-log budgets. Asset generated:
  `results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.svg`.
- Table 1: model-mediated packet rows.
- Table 2: deterministic source-destroying controls.
- Table 3: reviewer-risk controls.
- Table 4: systems bytes/tokens.
- Table 5: Qwen3 target-decoder smoke.
- Appendix table: per-seed exact IDs and artifact hashes.
