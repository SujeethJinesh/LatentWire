# Source Likelihood Syndrome Sidecar References

Date: 2026-04-27

## Blocker

The live branch needs a source-derived signal that is not another decoded-text
guard or target-cache effect. SVAMP70 live/holdout has source headroom, but
shallow source text features, source confidence routing, process repair, prefix
emitters, and query-memory variants have failed strict controls or disjoint
holdout. The specific design problem is therefore a rate-capped source signal
that can be decoded with target-side candidate context and falsified by
source-destroying controls.

## Primary Sources

- Slepian and Wolf, "Noiseless Coding of Correlated Information Sources,"
  IEEE Transactions on Information Theory, 1973. Problem helped: motivates
  transmitting only residual information when the decoder has correlated side
  information. Mechanism suggested: a compact syndrome-like signal instead of
  raw source text or hidden states. Role: theory support.

- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder," IEEE Transactions on Information Theory, 1976,
  DOI `10.1109/TIT.1976.1055508`.
  URL: https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  Problem helped: target has candidate/cache context that should act as decoder
  side information. Mechanism suggested: source should send a low-rate
  conditional sketch that is only useful when paired with the target-side
  candidate pool. Role: theory support and experiment framing.

- Pradhan and Ramchandran, "Distributed Source Coding Using Syndromes
  (DISCUS): Design and Construction," IEEE Transactions on Information Theory,
  2003. Problem helped: how to turn side-information coding into a concrete
  syndrome/code design. Mechanism suggested: send a compact bin/rank/syndrome
  rather than full source rationale. Role: inspiration for the current
  top-label plus quantized-margin sketch.

- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
  Image Encoders and Large Language Models," ICML 2023.
  URL: https://proceedings.mlr.press/v202/li23q/li23q.pdf
  Problem helped: learned fixed-size query bottlenecks can bridge frozen
  systems. Mechanism suggested: if the likelihood sketch fails, revive a
  query bottleneck only with target-conditioned decoding and strict source
  controls. Role: inspiration, not a fair baseline.

- Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
  Communication," ICLR 2023.
  URL: https://openreview.net/forum?id=SrC-nwieGJ
  Problem helped: cross-model latent mismatch and gauge freedom. Mechanism
  suggested: anchor-relative scores are safer than raw hidden-state transport
  when crossing model families. Role: inspiration for future latent-side
  retries if the current discrete sketch fails.

- Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation,"
  NeurIPS 2024.
  URL: https://arxiv.org/abs/2404.14469
  Problem helped: source signal should be query/candidate focused, not a bulk
  cache dump. Mechanism suggested: select/rank information before generation
  using a compact query-dependent criterion. Role: systems baseline/inspiration
  for byte and latency reporting.

- DeltaKV, "Residual-Based KV Cache Compression via Long-Range Similarity,"
  arXiv 2026.
  URL: https://arxiv.org/abs/2602.08005
  Problem helped: raw KV transport is too costly and fragile. Mechanism
  suggested: residual coding against shared or retrieved references. Role:
  adjacent compression inspiration; not a current baseline until a KV branch is
  revived.

- Zhang et al., "Query-Focused Retrieval Heads Improve Long-Context Reasoning
  and Re-ranking," EMNLP 2025.
  URL: https://aclanthology.org/2025.emnlp-main.1214/
  Problem helped: source information must be tied to the target query/candidate
  surface. Mechanism suggested: use query-focused scoring/selection rather than
  global source confidence. Role: inspiration for the source likelihood sketch.

- Thasarathan et al., "Universal Sparse Autoencoders: Interpretable
  Cross-Model Concept Alignment," ICML 2025.
  URL: https://openreview.net/forum?id=UoaxRN88oR
  Problem helped: interpretable cross-model alignment. Mechanism suggested:
  shared sparse concept codes as a later branch if the low-rate discrete
  likelihood sketch passes only weakly. Role: future branch inspiration.

## Experiment Impact

This literature update changes the next experiment from another shallow router
over decoded source text to a conditional syndrome-like sidecar:

1. Source encoder: source model scores a small target/text/source candidate pool
   by continuation likelihood.
2. Sidecar: transmit only the top candidate label plus a quantized confidence
   margin under a fixed bit budget.
3. Decoder: target falls back to target-alone unless a live-CV rule accepts the
   sidecar.
4. Controls: zero-source, shuffled-source, label-shuffle, target-only, and
   slots-only sketches must not recover clean source-only IDs.
5. Validation: tune only on `svamp70_live_source`, then freeze and test on
   `svamp70_holdout_source`.

Promotion does not change: the method must beat target/text relay on the
decision surface, preserve target-correct examples, clear source-destroying
controls, report bytes/latency, and become seed-stable before it can be a paper
claim.
