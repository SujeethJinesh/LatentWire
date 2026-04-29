# Source-Private Systems And Novelty Review

- date: `2026-04-29`
- status: strengthens full-paper framing; does not add a new headline claim
- live branch: source-private diagnostic packets with decoder side information

## Current Readiness

The evidence supports a scoped positive-method paper, but not yet a broad
latent-transfer or universal cross-model-communication paper. The strongest
ICLR path is to make four technical contributions explicit:

1. source-private communication with decoder side information;
2. a rate-capped diagnostic packet interface;
3. a strict source-destroying evaluation harness;
4. a systems rate/latency/compression analysis against text and cache-style
   communication baselines.

The largest remaining gap is not another small accuracy repeat. It is convincing
reviewers that the contribution is a useful communication interface and
evaluation standard, rather than a toy coded-label task.

## Story To Defend

The target has public problem context and candidate side information. The
source has private evidence from tools, logs, tests, memory, or hidden state.
The source may send only a rate-capped message. A method is successful only if
the gain disappears under zero-source, shuffled-source, random same-byte,
answer-only, answer-masked, target-derived, and matched-byte text controls.

This is a different point in the design space from cache transfer, activation
transfer, or prompt compression: it is source-private, interpretable, and
extreme-rate.

## Systems Evidence

The systems summary artifact is
`results/source_private_systems_summary_20260428/`.

Current headline:

- deterministic 2-byte diagnostic packets reach accuracy `1.000` on core and
  held-out `500`-example surfaces;
- matched-byte text relays stay at the target floor, with maximum accuracy
  `0.250`;
- full hidden-log relay is also `1.000`, but costs `366.45-373.50` bytes;
- the 2-byte packet is `183.2x-186.7x` smaller than full hidden-log relay and
  `7.0x` smaller than the full diagnostic text;
- model-produced packet rows report source latency and validity separately:
  Qwen3 p50 source latency about `293 ms`, Phi-3 about `422 ms`, Gemma 4 E2B
  large-slice MPS about `754 ms`;
- target-decoder rows are slower and remain ablations, but paired CPU n64
  core/held-out rows now pass at `0.656` and `0.719` matched accuracy versus
  target/control floors near `0.250`, reducing the hand-coded decoder
  objection.

This is a real systems contribution only if reported as a rate frontier, not as
a claim that text relay is inferior at all budgets. The fair statement is:
structured text becomes oracle once it has enough bytes to carry the diagnostic;
the packet wins at the far-left rate point while preserving source-destroying
controls.

## Related-Work Position

Primary comparison families:

- C2C and KVComm communicate high-dimensional KV/cache state. They are important
  baselines for internal-state communication, but they require model internals
  and occupy a higher-rate point.
- Activation communication exchanges hidden states between agents. It is closer
  to latent transfer than our current method, so the paper should not overclaim
  latent communication.
- LLMLingua compresses visible prompt/context. It is not a source-private
  residual communication method, but it motivates structured text compression
  controls.
- AutoGen, ReAct, Toolformer, and Chain-of-Agents motivate practical agent
  handoff via natural language or tool traces. Our contribution is a compact
  alternative when full trace relay is expensive or private.
- Wyner-Ziv and indirect source coding with decoder side information provide
  the cleanest theory frame: the packet only needs to encode the residual that
  the target cannot infer from public candidates.

## Reviewer Risks

1. **Coded-label objection.** The current method uses an interpretable packet
   protocol. The response is the codebook-remap gate, source-destroying
   controls, exact-ID parity, and rate-curve comparison, not a claim of
   universal semantics.
2. **Text relay objection.** Structured text is a strong baseline at larger
   budgets. The paper must show the rate curve and avoid saying text cannot
   solve the task.
3. **Protocol decoder objection.** The deterministic decoder is clean for
   isolating source-side communication. The Qwen3 target-decoder n64 row should
   be framed as an ablation, not the main method.
4. **External validity objection.** Gemma/Qwen/Phi evidence is strong; Granite
   is prompt-contract sensitive; OLMo fails. MoE/FP8 remains unclaimed until
   endpoint gates pass.
5. **Novelty objection.** The novelty is the combination of source-private
   threat model, decoder side information, explicit rate cap, source-destroying
   controls, and systems accounting. It is not the invention of all cross-model
   communication.

## Strongest Next Technical Contribution

The best one-month successor is a learned Wyner-Ziv / syndrome side-information
encoder over target candidate pools.

Bounded gate:

- source-private setup: target sees public question and candidates; source sees
  private diagnostic evidence;
- method: train a small encoder to emit a rate-capped binary/byte packet that
  selects or ranks candidates using target-side side information;
- controls: zero-source, shuffled-source, random same-byte, answer-only,
  answer-masked, target-derived sidecar, matched-byte structured text, and
  codebook remap;
- pass rule: learned packet beats target-only and matched-byte text by at least
  `15` points, controls remain within `2` points of target-only, and the learned
  code transfers to a held-out codebook or held-out family surface;
- runtime: feasible on MacBook if the first gate is a synthetic exact-ID
  candidate benchmark with frozen embeddings and a tiny MLP/linear syndrome
  encoder.

This would give the paper a less hand-designed second method contribution while
staying inside the same rigorous source-private evaluation contract.

## Decision

For the current paper package, keep the headline scoped to source-private
diagnostic packets. For the next technical contribution, implement the learned
syndrome packet gate before spending more time on additional source-emitter
repeats.
