# Reference Memo 720: Target-Aligned Feature Basis Scout

Date: 2026-05-04

## Local Status

LatentWire is not ICLR-ready. The current positive story is Sparse Resonance
Packets: low-rate, source-private, interpretable atom packets that improve a
frozen receiver under destructive controls. The current blocker is basis
quality. The first ARC sparse PCA packet is cleanly instrumented but fails
target-only, target-derived, atom-shuffle, same-byte text, and Qwen-substitution
controls.

## Primary Citations and Boundary Notes

- Raghu et al., 2017, "SVCCA: Singular Vector Canonical Correlation Analysis
  for Deep Learning Dynamics and Interpretability."
  Source:
  https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability
  Boundary: CCA/SVCCA is a shared-subspace diagnostic and initializer; it is not
  a causal communication method.

- Kornblith et al., 2019, "Similarity of Neural Network Representations
  Revisited."
  Source: https://proceedings.mlr.press/v97/kornblith19a.html
  Boundary: CKA is the reviewer-safe way to report representation similarity,
  but similarity alone cannot support the packet claim.

- Moschella et al., 2023, "Relative Representations Enable Zero-Shot Latent
  Space Communication."
  Source: https://openreview.net/forum?id=SrC-nwieGJ
  Boundary: anchor-relative coordinates are the strongest novelty risk for any
  anchor/common-basis claim; LatentWire must use them as a baseline/control or
  show a stricter downstream fixed-byte packet result.

- Chen et al., 2025, "Transferring Linear Features Across Language Models With
  Model Stitching."
  Source: https://arxiv.org/abs/2506.06609
  Boundary: affine residual-stream maps can transfer SAE features, probes, and
  steering vectors; this is a direct linear-map competitor for target-aligned
  basis packets.

- Grave, Joulin, and Berthet, 2019, "Unsupervised Alignment of Embeddings with
  Wasserstein Procrustes."
  Source: https://proceedings.mlr.press/v89/grave19a.html
  Boundary: OT plus orthogonal Procrustes is a compact alignment baseline for
  activation/atom clouds when paired supervision is weak.

- Lan et al., 2024/2025, "Quantifying Feature Space Universality Across Large
  Language Models via Sparse Autoencoders."
  Source: https://arxiv.org/abs/2410.06981
  Boundary: supports cross-model SAE feature-space overlap after matching by
  activation correlation, but it remains proxy similarity rather than causal
  receiver utility.

- Thasarathan et al., 2025/2026, "Universal Sparse Autoencoders: Interpretable
  Cross-Model Concept Alignment."
  Source: https://arxiv.org/abs/2502.03714
  Boundary: shared SAE dictionaries are a plausible packet atom source; the
  paper's reconstruction/alignment objective is not yet a strict downstream
  communication contract.

- Jiralerspong and Bricken, 2026, "Cross-Architecture Model Diffing with
  Crosscoders."
  Source: https://arxiv.org/abs/2602.11729
  Boundary: crosscoders isolate shared and model-unique features across
  architectures, directly matching the shared/private atom split needed for
  Sparse Resonance Packets.

- Paulo, Shabalin, and Belrose, 2025, "Transcoders Beat Sparse Autoencoders for
  Interpretability."
  Source: https://arxiv.org/abs/2501.18823
  Boundary: transcoders may produce more behavior-relevant atoms than vanilla
  activation-reconstruction SAEs, especially when the packet must affect a
  downstream component.

- Karvonen et al., 2025, "SAEBench: A Comprehensive Benchmark for Sparse
  Autoencoders in Language Model Interpretability."
  Source: https://arxiv.org/abs/2503.09532
  Boundary: proxy SAE metrics can fail to predict practical utility; LatentWire
  should rank bases by paired downstream ARC improvement and destructive
  controls, not reconstruction loss.

- Fu et al., 2026, "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models."
  Source: https://openreview.net/forum?id=LeatkxrBCi
  Boundary: dense KV-cache projection/fusion is the strongest direct semantic
  communication competitor; LatentWire novelty must be utility per byte,
  source privacy, and interpretability under controls.

- Ramesh and Li, 2025, "Communicating Activations Between Language Model
  Agents."
  Source: https://arxiv.org/abs/2501.14082
  Boundary: activation communication is an important non-text competitor, but
  it lacks the fixed-byte sparse atom packet and destructive-control contract.

## Actionable Next Mac-First Experiments

1. CCA-whitened sparse packet:
   fit paired source/target hidden public innovations on ARC train/validation
   disagreement rows, whiten with regularized CCA, send top-k source canonical
   coordinates with quantized signs/magnitudes, and decode through the existing
   Qwen soft-prefix receiver. Controls: atom shuffle, coefficient shuffle,
   source-row shuffle, candidate roll, target-derived CCA coordinates, and
   source-score/rank packets.

2. Target-Procrustes packet:
   learn a ridge or orthogonal Procrustes map from TinyLlama candidate hidden
   innovations into Qwen hidden innovations, then sparse-code in the target PCA
   or target SAE basis rather than source PCA. This directly tests whether the
   current PCA failure is a source-basis problem.

3. Anchor-relative residual packet:
   build an answer-key-forbidden anchor bank from target prompts/candidates,
   represent source candidate innovations by similarities to the target anchor
   bank, then transmit a top-k sparse residual over anchors. This is the
   required baseline against relative-representation novelty.

4. Tiny crosscoder atom packet:
   train a small shared/private crosscoder on paired ARC candidate states.
   Packet only shared atoms plus source-private atoms whose train-only
   calibration predicts positive target benefit. Compare shared-only,
   private-only, shared+private, and atom-knockout rows.

5. Transcoder-style behavior packet:
   train a small input-to-output transcoder from source candidate hidden input
   to target-side prefix delta or target logit-margin delta. If it beats SAE/PCA
   at the same bytes, promote behavior-relevant atoms over reconstruction atoms.

## Reviewer Risks

- Relative representations and model stitching can absorb any broad "common
  latent space" claim. The contribution must be fixed-byte, source-private,
  causally useful packets, not alignment.
- CCA/CKA/SVCCA are diagnostics. A high-alignment plot without paired task
  improvement and destructive-control separation will read as proxy chasing.
- SAE/crosscoder/transcoder atoms are not automatically interpretable or useful.
  Rank them by held-out target improvement, atom identity causality, and
  coefficient causality.
- C2C and activation communication own the broad "LLMs communicate through
  activations/KV" territory. LatentWire must win on utility per byte, privacy,
  auditability, and Mac-to-native reproducibility.
- The current n8 gate is too small for claims. Use it only as a falsification
  surface; widen after a method beats target-only, target-derived,
  source-family substitution, atom shuffle, coefficient shuffle, wrong-row,
  candidate-roll, source-rank/score, and same-byte controls.

## Decision

The highest-value next branch is not a larger sparse PCA packet. It is a
target-aligned basis packet: CCA-whitened or Procrustes-to-target coordinates
first, then a small shared/private crosscoder only if the linear target-aligned
packet shows atom identity causality on the strict ARC n8/n16 gate.
