# Byte-Efficient Source Side-Information References

Date: 2026-04-27

## Status

This memo updates the reference stack after the historical MD/result audit and
the KVComm control-harness work. The blocker is no longer whether latent or
cache communication is a plausible paper topic; recent baselines already make
that case. The blocker is whether this repo can produce a source-derived,
target-preserving method that survives source-destroying controls while using
fewer bytes than raw KV/cache transfer.

## Sources and Experiment Implications

### Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression

- Primary source: `https://arxiv.org/abs/2512.17914`
- Problem it helps with: raw KVComm is byte-heavy in the local smoke
  (`530432` communicated bytes/example at a 0.25 layer fraction).
- Mechanism/design idea: adaptive layer-wise quantization plus calibration
  across heterogeneous models suggests that the fair systems baseline is not
  only selective KV layers, but selective plus quantized KV.
- Does it change the next experiment? Yes. The next KV baseline should include
  a byte-normalized or quantized control before claiming systems value for a
  new sidecar.
- Role: baseline and systems comparator.

### DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving

- Primary source: `https://arxiv.org/abs/2411.02820`
- Problem it helps with: separating same-architecture KV reuse from true
  cross-family communication.
- Mechanism/design idea: reuse compatible KV layers and selectively recompute
  fragile layers, with throughput and TTFT as first-class metrics.
- Does it change the next experiment? Yes. Same-base or same-architecture cache
  reuse must be reported separately from cross-family transfer, and any systems
  claim should include prefill/TTFT where practical.
- Role: baseline and threat model.

### KVComm: Enabling Efficient LLM Communication through Selective KV Sharing

- Primary source: `https://arxiv.org/abs/2510.03346`
- Problem it helps with: reviewer-facing comparison against direct cache
  sharing rather than only text relay.
- Mechanism/design idea: layer selection by attention importance is a
  structured baseline for any source-state transfer method.
- Does it change the next experiment? Yes. Keep `latent_bridge.kvcomm_eval`
  as the fixed-budget cache baseline, but require matched/zero/shuffled/
  target-only controls and byte telemetry.
- Role: baseline, ablation, and systems accounting.

### Cache-to-Cache: Direct Semantic Communication Between Large Language Models

- Primary source: `https://arxiv.org/abs/2510.03215`
- Problem it helps with: direct semantic communication through target cache
  fusion is the main quality competitor to a new latent side-information
  method.
- Mechanism/design idea: preserve the target cache while fusing projected
  source KV information; this supports zero-init receiver gates and
  target-self-preservation controls.
- Does it change the next experiment? Yes. A new method must either compete
  with C2C on quality or beat it on bytes/latency/TTFT at comparable accuracy.
- Role: baseline and paper framing.

### Latent Space Communication via K-V Cache Alignment

- Primary source: `https://arxiv.org/abs/2601.06123`
- Problem it helps with: cross-model mismatch and raw RotAlign instability.
- Mechanism/design idea: adapters into a shared cache representation are a
  cleaner version of the repo's historical RotAlign/DynAlign attempts, but the
  local lesson is to combine this with target-preserving gates and source
  controls.
- Does it change the next experiment? It weakly revives alignment only as
  target-safe shared-space side information, not raw transported cache.
- Role: inspiration and baseline family.

### Semantic Communication Between Agents

- Primary source: `https://openreview.net/forum?id=d68YsO4KsB`
- Problem it helps with: paper framing for agent-to-agent communication beyond
  text.
- Mechanism/design idea: communication should be evaluated by task utility and
  semantic sufficiency, which matches the repo's exact-ID candidate-pool and
  source-destroying-control gates.
- Does it change the next experiment? No direct code change, but it supports
  framing side-information as task-conditioned communication, not hidden-state
  reconstruction for its own sake.
- Role: paper framing and theory support.

### Slepian-Wolf / Wyner-Ziv Side-Information Coding

- Primary sources:
  - `https://link.springer.com/article/10.1155/ASP.2005.961`
  - `https://oaktrust.library.tamu.edu/handle/1969.1/2751?show=full`
  - `https://arxiv.org/abs/2106.02797`
- Problem it helps with: source signal is useful only relative to target-side
  context; sending full state wastes bytes and invites target-cache artifacts.
- Mechanism/design idea: source should send a syndrome/innovation/residual that
  is decoded using target-side side information, with zero/random/shuffled
  syndrome controls.
- Does it change the next experiment? Yes. The next CPU/MPS method branch
  should be an erasure-aware learned syndrome or innovation sidecar, not
  another shallow confidence router.
- Role: theory support and experiment design.

Neural distributed source coding adds the practical learned-code version of
this principle: train an encoder/decoder for complex correlations when side
information is only available at the decoder. Locally, that means the target
candidate/cache state should be treated as decoder side information, and the
source message should be evaluated as a rate-distortion curve rather than a
single cherry-picked byte point.

### Universal Sparse Autoencoders / Cross-Model Concept Alignment

- Primary source: `https://openreview.net/forum?id=UoaxRN88oR`
- Problem it helps with: raw geometric alignment is unstable and hard to
  interpret.
- Mechanism/design idea: use shared sparse concept dictionaries or
  anchor-relative atoms as a byte-efficient, interpretable sidecar: source can
  transmit atom IDs and weights instead of full dense states.
- Does it change the next experiment? It is a second-line branch after a
  stronger source surface clears: anchor-relative sparse difference atoms with
  private-atom zeroing and same-sparsity shuffled controls.
- Role: inspiration, interpretability, and ablation design.

### Cross-Architecture Model Diffing with Crosscoders

- Primary source: `https://arxiv.org/abs/2602.11729`
- Problem it helps with: historical RotAlign/DynAlign results are not
  interpretable enough, and cross-family mismatch remains a major failure mode.
- Mechanism/design idea: dedicated-feature crosscoders can isolate shared and
  model-specific features across architectures, suggesting a diagnostic for
  whether communicated latents are shared reasoning features or source-private
  artifacts.
- Does it change the next experiment? Not before the next positive smoke. If a
  sidecar clears the small gate, add a crosscoder/dictionary diagnostic for
  transmitted atoms.
- Role: interpretability inspiration and post-gate diagnostic.

## Next Branch Update

The next positive-method branch should be framed as source side-information for
a target-side decoder, not raw cache transport:

1. Source encoder: emit a tiny learned syndrome/innovation over target
   candidates or sparse source-difference atoms.
2. Decoder: use target candidates/cache as side information; abstain when the
   source message is not decisive.
3. Controls: zero source, shuffled source, random sidecar, target-only,
   slots-only, same-byte random sidecar, and source-answer overlap checks.
4. Systems comparison: KVComm, Q-KVComm-style quantized KV, C2C, text relay,
   and target self-repair.

Promotion gate: on a stronger exact-ID surface, matched side-information must
recover clean source-necessary examples that no source-destroying control
recovers, preserve target-correct examples, and use far fewer bytes than raw
KVComm.
