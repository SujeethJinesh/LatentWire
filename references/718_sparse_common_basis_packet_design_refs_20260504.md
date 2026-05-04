# Reference Memo 718: Sparse Common-Basis Packet Design

Date: 2026-05-04

## Local Decision Boundary

The current LatentWire gate is not blocked by absence of source headroom. It is
blocked by the receiver failing to safely identify which source-private evidence
is useful. Recent ARC and HellaSwag gates demote shallow score packets and
single-step hidden soft-prefix encoders, but keep common-basis source features
alive if they are tested under strict destructive controls.

The next common-basis method should treat sparse atom IDs and coefficients as a
causal packet, not as an interpretability claim. A positive result must improve
the frozen target beyond target-only/fixed-hybrid and beat atom-shuffle,
wrong-row, source-rank/score, target-derived, same-byte, and source-family
substitution controls.

## Primary Sources

- Bricken et al., 2023, "Towards Monosemanticity: Decomposing Language Models
  With Dictionary Learning."
  Source: https://transformer-circuits.pub/2023/monosemantic-features/
  Boundary: establishes SAE/dictionary-learning features as better units than
  raw neurons, but does not solve cross-model communication or causal receiver
  usefulness.

- Templeton et al., 2024, "Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet."
  Source: https://transformer-circuits.pub/2024/scaling-monosemanticity/
  Boundary: shows SAEs can scale and yield steerable/interpretable features,
  but remains single-model interpretability rather than source-private packets.

- Lan et al., 2024, "Sparse Autoencoders Reveal Universal Feature Spaces Across
  Large Language Models."
  Source: https://arxiv.org/abs/2410.06981
  Boundary: supports cross-model feature overlap after matching SAE neurons by
  activation correlation and representation-similarity metrics.

- "Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment,"
  2025.
  Source: https://arxiv.org/abs/2502.03714
  Boundary: directly motivates a shared sparse concept space across models, but
  still optimizes reconstruction/alignment rather than downstream target repair.

- Jiralerspong and Bricken, 2026, "Cross-Architecture Model Diffing with
  Crosscoders."
  Source: https://arxiv.org/abs/2602.11729
  Boundary: crosscoders can separate shared/private features across
  architectures, which is useful for packet atom selection; model diffing is not
  a communication protocol.

- "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning
  Regimes," 2026.
  Source: https://arxiv.org/abs/2603.04426
  Boundary: delta-weighted crosscoders motivate behavior-changing atom
  selection, especially for source innovations.

- Paulo, Shabalin, and Belrose, 2025, "Transcoders Beat Sparse Autoencoders for
  Interpretability."
  Source: https://arxiv.org/abs/2501.18823
  Boundary: transcoders reconstruct component outputs from inputs and may find
  more behavior-relevant atoms than vanilla SAE reconstruction.

- Karvonen et al., 2025, "SAEBench: A Comprehensive Benchmark for Sparse
  Autoencoders in Language Model Interpretability."
  Source: https://arxiv.org/abs/2503.09532
  Boundary: warns that SAE proxy metrics do not reliably transfer to practical
  utility; LatentWire should evaluate causal task improvement, not just
  reconstruction or interpretability.

- Moschella et al., 2023, "Relative Representations Enable Zero-Shot Latent
  Space Communication."
  Source: https://openreview.net/forum?id=SrC-nwieGJ
  Boundary: anchor-relative coordinates are a strong common-coordinate baseline
  and novelty risk; use them as controls/initializers, not as the claimed
  contribution.

- Raghu et al., 2017, "SVCCA: Singular Vector Canonical Correlation Analysis
  for Deep Learning Dynamics and Interpretability."
  Source: https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability
  Boundary: CCA/SVCCA/CKA/Procrustes can diagnose shared subspaces but do not
  by themselves establish causal communication.

- Chen et al., 2025, "Transferring Features Across Language Models With Model
  Stitching."
  Source: https://arxiv.org/abs/2506.06609
  Boundary: learned maps can transfer SAE features across models; this is a
  baseline for shared-basis transfer, not a fixed-byte packet contract.

- Park, Choe, and Veitch, 2023, "The Linear Representation Hypothesis and the
  Geometry of Large Language Models."
  Source: https://arxiv.org/abs/2311.03658
  Boundary: concept directions and steering depend on the right inner product;
  naive Euclidean atom matching is a likely failure mode.

- Turner et al., 2023, "Steering Language Models With Activation Engineering."
  Source: https://arxiv.org/abs/2308.10248
  Boundary: activation additions establish causal hidden-state control, but
  dense steering vectors are not cross-model sparse packet evidence.

- Fu et al., 2026, "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models."
  Source: https://openreview.net/forum?id=LeatkxrBCi
  Boundary: strongest direct communication competitor; exposes dense KV caches
  rather than fixed-byte sparse atom packets.

- Ramesh and Li, 2025, "Communicating Activations Between Language Model
  Agents."
  Source: https://arxiv.org/abs/2501.14082
  Boundary: activation communication can beat text in some settings, but lacks
  LatentWire's fixed-byte/source-private/destructive-control contract.

- Li et al., 2023, "BLIP-2: Bootstrapping Language-Image Pre-training with
  Frozen Image Encoders and Large Language Models."
  Source: https://arxiv.org/abs/2301.12597
  Boundary: Q-Former/query-token bottlenecks are a strong connector template
  for turning foreign representations into target-consumable soft slots.

## What Looks Solved

- Sparse feature extraction is mature enough to be a reasonable basis family:
  SAE, universal SAE, crosscoder, and transcoder variants are all plausible
  atom generators.
- Cross-model overlap is plausible but partial. Existing evidence supports
  matched feature spaces and shared/private decomposition; it does not prove
  task-useful packet transfer.
- Linear/CCA/PCA/Procrustes/relative-coordinate methods are useful diagnostics,
  initializers, and baselines, but are not enough for a positive method claim.
- Activation steering proves hidden directions can causally affect behavior;
  LatentWire still needs source-conditioned per-example atom packets.

## What Is Not Solved

- Selecting atoms that encode source-specific innovation rather than source
  answer rank, candidate order, or target-cache artifacts.
- Making sparse atom IDs portable across model families without using hidden
  dense vectors as the actual channel.
- Showing downstream receiver improvement at fixed bytes, not reconstruction
  loss, feature similarity, or interpretability alone.
- Separating common semantic atoms from private family/style atoms and deciding
  when private atoms should be withheld, translated, or used as uncertainty.

## Mac-First Method Designs

1. Shared SAE packet to target-native soft slots.
   Fit a small universal SAE on answer-key-forbidden candidate hidden states
   from TinyLlama/Qwen/Phi on ARC or HellaSwag. Source sends top-k shared atom
   IDs plus quantized coefficients. A tiny receiver maps atom packets to Qwen or
   Phi soft prefixes. Sweep k and coefficient bits. Gate on held-out
   disagreement rows.

2. Shared/private crosscoder innovation packet.
   Train a paired crosscoder on source and target candidate hidden states, with
   shared atoms reconstructing both and private atoms reconstructing residuals.
   Packet includes only source atoms whose target-side predicted benefit is
   positive under train-only calibration. Compare shared-only, private-only,
   shared+innovation, and target-derived packet controls.

3. Anchor-relative sparse residual code.
   Build a frozen anchor bank from target-only prompts and compute source
   hidden states in anchor-relative coordinates before sparse coding. Transmit
   sparse residual atoms relative to the target's own cache/state estimate. This
   tests whether quotient/relative coordinates make atom IDs less
   family-specific while keeping the packet compact.

## Required Controls

- Atom-ID shuffle with coefficient distribution preserved.
- Coefficient shuffle with atom IDs preserved.
- Wrong-row source packet and within-batch derangement.
- Candidate-roll packet for MCQ tasks.
- Source-rank/source-index/source-score packets at matched bytes.
- Target-derived packet generated from target states only.
- Zero-source and same-byte random packet.
- Source-family substitution, e.g. Qwen packet in place of TinyLlama packet
  when testing TinyLlama-to-Qwen ARC.
- Reconstruction-only positive control: if reconstruction improves but task
  accuracy does not, demote the basis.
- Oracle atom upper bound: choose between target-only and packet receiver per
  row to quantify headroom before more method work.

## Decision

Promote sparse/common-basis packets as the next ARC soft-prefix source encoder
only if the first Mac gate is small and destructive-control-heavy. Do not claim
feature universality as the contribution. The contribution, if it works, is a
fixed-byte causal packet whose atom IDs and coefficients improve a frozen
target under controls that destroy source identity, candidate order, score-rank
leakage, and target-cache shortcuts.
