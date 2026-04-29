# Masked Innovation Receiver References

- date: `2026-04-29`
- blocker: the learned receiver is same-distribution positive but fails or
  becomes asymmetric across held-out families.
- next experiment affected: sparse/shared-dictionary innovation receiver only;
  the current masked innovation projection is not promoted after the first
  cross-family failure.

## Sources

1. I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture
   (`https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf`).
   - blocker helped: motivates predicting masked latent representations rather
     than reconstructing text or answers.
   - mechanism/design idea: train the receiver to predict target-side
     innovation in representation space.
   - changed experiment: yes; source packet became matched-minus-answer-masked
     source innovation.
   - use: objective inspiration.

2. Flow Matching for Generative Modeling (`https://arxiv.org/abs/2210.02747`).
   - blocker helped: reframes source-to-target repair as a conditional
     residual/velocity rather than static alignment.
   - mechanism/design idea: predict one-step velocity from target-prior
     candidate representation to answer-candidate representation.
   - changed experiment: yes; the receiver scores candidate innovations against
     the packet.
   - use: method inspiration.

3. Relative Representations Enable Zero-Shot Latent Space Communication
   (`https://openreview.net/forum?id=SrC-nwieGJ`).
   - blocker helped: raw latent/candidate coordinates are not invariant across
     families.
   - mechanism/design idea: encode candidates by similarity to shared anchors.
   - changed experiment: yes; packets target anchor-relative innovations.
   - use: method design and cross-family diagnostic.

4. Sparse Crosscoders for Cross-Layer Features and Model Diffing
   (`https://arxiv.org/abs/2603.05805`).
   - blocker helped: distinguishes shared versus model-specific sparse
     features.
   - mechanism/design idea: the next variant should use shared/unique sparse
     dictionaries and feature knockout, not just ridge projection.
   - changed experiment: not yet; this becomes the next gate after the current
     failure.
   - use: next-method design and interpretability.

5. Anthropic Crosscoder Model Diffing
   (`https://www.anthropic.com/research/crosscoder-model-diffing`).
   - blocker helped: cautions against treating sparse features as aligned
     semantics without ablations.
   - mechanism/design idea: require feature knockout and dead-feature/usage
     telemetry before claiming interpretability.
   - changed experiment: sets promotion criteria for the next variant.
   - use: framing and ablation design.

## Outcome

The resulting masked innovation receiver passed a same-distribution smoke but
failed core-to-holdout despite oracle headroom of `1.000`. This means the code
space is expressive enough, but the learned source-private innovation map is
not cross-family transferable. The literature still supports the branch only
after adding shared-dictionary/crosscoder calibration and knockout controls.
