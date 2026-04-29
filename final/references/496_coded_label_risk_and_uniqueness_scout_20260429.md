# Coded-Label Risk And Uniqueness Scout

- date: `2026-04-29`
- blocker: reviewers can still argue that the strongest LatentWire result is a
  synthetic coded-label lookup rather than a source-private communication
  method with decoder side information.
- role: primary-source scout memo for novelty framing and next baselines after
  the coded-label stress gate.

## Sources And Experiment Implications

1. C2C: Cache-to-Cache (`https://arxiv.org/abs/2510.03215`).
   - blocker helped: direct novelty threat for non-text LLM communication.
   - mechanism/design idea: cache projection/fusion is a high-rate internal
     state handoff baseline.
   - next experiment change: compare on byte/model-access/latency axes, not as
     a same-byte baseline.
   - use: baseline and framing.

2. KVComm: Selective KV Sharing (`https://openreview.net/forum?id=F7rUng23nw`).
   - blocker helped: positions KV sharing as the mature alternative to compact
     packets.
   - mechanism/design idea: select informative KV pairs or layers for transfer.
   - next experiment change: keep KV/cache payload lower-bound rows in the
     systems table and avoid claiming generic cache-compression superiority.
   - use: baseline.

3. TurboQuant (`https://arxiv.org/abs/2504.19874`).
   - blocker helped: prevents weak quantization comparisons.
   - mechanism/design idea: random rotation plus scalar quantization and
     QJL-style residuals motivate compression-native packet controls.
   - next experiment change: add TurboQJL-style matched-byte residual baselines
     only for learned/vector branches, not for the scalar diagnostic packet.
   - use: baseline and ablation.

4. QJL (`https://arxiv.org/abs/2406.03482`).
   - blocker helped: random sign sketches are a strong low-bit similarity
     baseline.
   - mechanism/design idea: JL projection plus one-bit signs can preserve inner
     products at low rate.
   - next experiment change: use sign-sketch candidate-similarity controls for
     any future latent/vector communication branch.
   - use: baseline and theory support.

5. D-JEPA (`https://openreview.net/forum?id=d4njmzM7jf`).
   - blocker helped: latent predictive objectives are relevant inspiration but
     not direct prior art for source-private packets.
   - mechanism/design idea: predict target-compatible latent state rather than
     reconstructing source evidence.
   - next experiment change: weak; it supports a future learned receiver loss,
     but does not alter the immediate coded-label risk gate.
   - use: inspiration.

6. Diffusion Transformers with Representation Autoencoders
   (`https://arxiv.org/abs/2510.11690`).
   - blocker helped: argues that decoder-compatible representation spaces are
     central for latent transport.
   - mechanism/design idea: representation-rich latents need compatible
     decoders; raw low-dimensional bottlenecks are not enough.
   - next experiment change: no immediate gate change; useful for framing why
     simple masked innovation failed cross-family.
   - use: framing and inspiration.

7. Distributed Indirect Source Coding with Decoder Side Information
   (`https://arxiv.org/abs/2405.13483`).
   - blocker helped: source-private communication with decoder side information
     has classical theory roots.
   - mechanism/design idea: recover a task variable with compact messages,
     rather than reconstructing the full source.
   - next experiment change: report rate/frontier and destructive controls as
     the empirical novelty, not side information itself.
   - use: theory and framing.

8. Slepian-Wolf and Wyner-Ziv source coding
   (`https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources`,
   `https://www.sciencedirect.com/science/article/pii/S0019995878900347`).
   - blocker helped: clarifies that syndrome-style coding is prior art.
   - mechanism/design idea: encoder sends compact information; decoder uses
     side information to resolve the source/task variable.
   - next experiment change: keep the paper claim scoped to LLM/tool-trace
     instantiation with source-destroying controls and systems accounting.
   - use: theory support.

## Direct Overlap Verdict

No source above directly overlaps the precise LatentWire claim: a
source-private, extreme-rate evidence packet decoded against target candidate
side information, with zero/shuffled/random/answer-only/answer-masked/target
sidecar controls. The nearest theory is Slepian-Wolf/Wyner-Ziv and distributed
indirect source coding. The nearest LLM systems work is C2C/KVComm, but those
move high-dimensional cache/internal state rather than two-byte task evidence.

## Next Experiment Decision

The highest-value Mac-local gate is still coded-label/protocol robustness:
candidate-label renaming, diagnostic-code remapping, candidate-order
permutation, and a composed stress row. If that passes, the next reviewer-facing
artifact should be a one-command reproduction bundle plus a novelty matrix
against C2C/KVComm, prompt compression, KV/cache quantization, and source coding.
