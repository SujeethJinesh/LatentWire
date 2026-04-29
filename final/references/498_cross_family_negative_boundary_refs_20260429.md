# Cross-Family Negative Boundary References

- date: `2026-04-29`
- blocker: the paper needs an honest boundary around failed/asymmetric
  cross-family learned communication while preserving motivation for a future
  shared-dictionary method.
- role: source memo for `source_private_cross_family_negative_boundary_20260429`.

## Local Evidence Sources

1. Candidate embedding receiver summaries:
   `results/source_private_candidate_embedding_receiver_20260429/`.
   - blocker helped: shows same-distribution learned receiver positives do not
     transfer cleanly to held-out families.
   - mechanism implication: dense/ridge and code-similarity packet decoders are
     not enough.
   - next experiment change: require shared sparse atoms and feature knockout.
   - use: negative boundary and ablation.

2. Learned Wyner-Ziv cross-family gate:
   `results/source_private_wyner_ziv_cross_family_gate_20260429/`.
   - blocker helped: establishes bidirectional cross-family scalar syndrome is
     failed/asymmetric.
   - mechanism implication: same-family/remap WZ positives do not automatically
     become family-invariant.
   - next experiment change: do not promote one-direction holdout-to-core rows.
   - use: negative boundary.

3. Masked innovation receiver:
   `results/source_private_masked_innovation_receiver_20260429/`.
   - blocker helped: shows high oracle headroom with failed learned packet
     transfer.
   - mechanism implication: source information is representable, but simple
     masked innovation maps do not transfer cross-family.
   - next experiment change: use shared dictionary/crosscoder decomposition
     rather than another dense projection.
   - use: negative boundary and future-method motivation.

4. Anchor-relative sparse packet smoke:
   `results/anchor_relative_sparse_packet_gate_20260429_smoke/`.
   - blocker helped: rules out shallow relative-coordinate sparse packets as a
     bidirectional cross-family fix.
   - mechanism implication: relative coordinates are useful but not sufficient
     without learned shared/private feature separation.
   - next experiment change: require feature knockout and bidirectional gates.
   - use: negative boundary.

## Primary-Source Motivation For The Next Method

1. Relative Representations
   (`https://openreview.net/forum?id=SrC-nwieGJ`).
   - blocker helped: cross-model coordinates are not naturally aligned.
   - mechanism idea: communicate relational features rather than raw latent
     coordinates.
   - next experiment change: shared dictionary should use relative/shared views.
   - use: theory and method inspiration.

2. Sparse Crosscoders (`https://arxiv.org/abs/2603.05805`) and Anthropic
   crosscoder model diffing (`https://www.anthropic.com/research/crosscoder-model-diffing`).
   - blocker helped: need shared-vs-private sparse feature decomposition.
   - mechanism idea: decompose source/target innovation into shared atoms and
     model/family-private atoms; transmit only shared atoms.
   - next experiment change: build shared sparse crosscoder knockout packet.
   - use: method inspiration and interpretability.

3. Universal/shared SAEs (`https://arxiv.org/abs/2502.03714`).
   - blocker helped: supports the possibility of reusable sparse features
     across model families.
   - mechanism idea: learn sparse shared features rather than dense projections.
   - next experiment change: track dead atoms, support entropy, and feature
     knockout effects.
   - use: method inspiration.

## Decision

Do not claim cross-family communication from the current rows. The next
cross-family method must be a shared sparse dictionary/crosscoder packet with
feature knockout, not another variant of dense masked innovation or static
anchor-relative scoring.
