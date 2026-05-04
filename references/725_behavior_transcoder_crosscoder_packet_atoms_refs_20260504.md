# Behavior-Transcoder And Shared-Private Crosscoder Packet Atom References

Date: 2026-05-04

## Current Gate Read

- Paper readiness: ICLR is still blocked. Sparse Resonance has a clean
  packet/byte interface, but source-only PCA and target-aligned PCA packets
  fail the strict ARC scouting surface.
- Current story: LatentWire_v2 should test low-rate, source-private,
  interpretable packet atoms whose meaning is tied to target answer behavior,
  not just hidden-state reconstruction.
- Exact blocker: a packet atom basis must beat target-only, target-derived,
  source-index/rank/score, same-byte text/random, source-row shuffle,
  candidate-roll, atom/coeff shuffle, top-atom knockout, and source-family
  substitution with paired uncertainty.

## Fresh Primary Sources

1. Sparse Crosscoders for Cross-Layer Features and Model Diffing.
   https://transformer-circuits.pub/2024/crosscoders/index.html
   - Use: motivates one sparse feature set with per-model decoder vectors.
   - Packet implication: transmit shared atom IDs only if shared decoders are
     target-readable; transmit private/delta atom bits only if they causally
     repair target answer margins.
   - Boundary: crosscoder features are not automatically causal for a target
     answer. Circuit/patch tests are required.

2. Insights on Crosscoder Model Diffing.
   https://transformer-circuits.pub/2025/crosscoder-diffing/index.html
   - Use: motivates explicit shared-feature and standard/private-feature
     partitions, with reduced sparsity penalty for shared features.
   - Packet implication: avoid a single undifferentiated sparse dictionary;
     reserve packet fields for shared target-readable atoms plus private
     source-correction atoms.
   - Boundary: model-exclusive features can be denser and more polysemantic.

3. Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning.
   https://arxiv.org/abs/2504.02922
   - Use: BatchTopK and Latent Scaling reduce artifacts in crosscoder diffing.
   - Packet implication: require latent-scaling diagnostics before calling an
     atom source-private or target-exclusive.
   - Boundary: the positive examples are base-vs-chat model diffing, not
     source-to-target MCQ communication.

4. Cross-Architecture Model Diffing with Crosscoders.
   https://openreview.net/forum?id=YXB8uigyOg
   - Use: direct evidence that crosscoders can expose behavioral differences
     across non-identical architectures.
   - Packet implication: this is the strongest source for a strict
     cross-family shared/private atom scout.
   - Boundary: submitted ICLR 2026 work, and still a diffing method rather
     than a communication protocol.

5. Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning
   Regimes. https://arxiv.org/abs/2603.04426
   - Use: delta-weighted crosscoder loss explicitly prioritizes directions
     that change behavior and reports causal mitigation.
   - Packet implication: train packet atoms on target margin deltas or
     source-only repair buckets, not reconstruction MSE.
   - Boundary: narrow fine-tuning model-organism setting; LatentWire needs
     independent source/target model pairs and source-private packets.

6. Transcoders Find Interpretable LLM Feature Circuits.
   https://arxiv.org/abs/2406.11944
   - Use: transcoders approximate dense MLP computation with sparse features
     and support circuit analysis through MLPs.
   - Packet implication: behavior-oriented atoms should be trained against
     component input-output behavior or target logit-margin effects.
   - Boundary: not a cross-model communication paper.

7. Circuit Tracing: Revealing Computational Graphs in Language Models.
   https://transformer-circuits.pub/2025/attribution-graphs/methods.html
   - Use: cross-layer transcoders form replacement models and attribution
     graphs for prompt-level behavior.
   - Packet implication: atom selection can be restricted to graph nodes on
     high-attribution paths to the target answer margin.
   - Boundary: replacement-model faithfulness is imperfect and must be logged.

8. Sparse Feature Circuits: Discovering and Editing Interpretable Causal
   Graphs in Language Models. https://arxiv.org/abs/2403.19647
   - Use: sparse feature circuits are explicitly causal and useful for
     downstream interventions.
   - Packet implication: top packet atoms need ablation/amplification effects
     on the downstream candidate decision, not only interpretability labels.
   - Boundary: circuits are within one model unless LatentWire adds a
     cross-model packet contract.

9. Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment.
   https://arxiv.org/abs/2502.03714
   - Use: shared sparse concept spaces can reconstruct multiple models.
   - Packet implication: useful baseline for a shared atom dictionary.
   - Boundary: primarily vision/cross-model concept alignment, not LLM MCQ
     answer repair.

10. Quantifying Feature Space Universality Across Large Language Models via
    Sparse Autoencoders. https://arxiv.org/abs/2410.06981
    - Use: supports the hypothesis that SAE feature spaces across LLMs may be
      analogous under transformations.
    - Packet implication: feature-space matching is a baseline.
    - Boundary: rotation-invariant similarity is not causal communication.

11. SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and
    Cross-Modal Interpretability. https://arxiv.org/abs/2507.06265
    - Use: Global TopK plus cross-reconstruction explicitly aligns latent IDs.
    - Packet implication: atom-ID alignment can be trained directly, then
      attacked with atom-ID derangement and top-atom knockout.
    - Boundary: vision/multimodal evidence, not target-answer causality.

12. Sparse Autoencoders Do Not Find Canonical Units of Analysis.
    https://arxiv.org/abs/2502.04878
    - Use: major threat model against treating SAE features as atomic or
      canonical.
    - Packet implication: use causal knockout, meta-feature/stability checks,
      and seed repeats before claiming packet atoms are meaningful.

13. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language
    Model Interpretability. https://arxiv.org/abs/2503.09532
    - Use: warns against proxy-only SAE evaluation.
    - Packet implication: report task utility, disentanglement/stability, and
      practical intervention metrics together.

14. Transferring Linear Features Across Language Models With Model Stitching.
    https://arxiv.org/abs/2506.06609
    - Use: affine residual maps can transfer SAE weights/probes/steering
      vectors between small and large LMs.
    - Packet implication: a cheap transferred-SAE initialization is Mac-feasible.
    - Boundary: feature transfer differs by semantic/structural/functional
      class and still requires downstream packet controls.

15. Relative representations enable zero-shot latent space communication.
    https://arxiv.org/abs/2209.15430
    - Use: anchor-relative coordinates handle gauge/isometry instability.
    - Packet implication: keep as a common-coordinate baseline.
    - Boundary: current PCA/relative-coordinate failures mean this is not the
      next highest-value branch unless paired with behavior-trained atoms.

16. Qwen-Scope SAE checkpoints.
    https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100
    - Use: official/pretrained Qwen residual-stream SAE artifacts that may make
      a target-side atom scout cheap.
    - Boundary: model mismatch and model-card claims do not prove LatentWire
      source-private communication.

17. Qwen-3 Transcoder checkpoints.
    https://huggingface.co/collections/mwhanna/qwen-3-transcoders
    - Use: pretrained transcoders for Qwen-3 models from 0.6B to 14B can make a
      target behavior-atom scout feasible on Mac.
    - Boundary: third-party artifact; validate checkpoint/model compatibility
      and replacement-model faithfulness before citing result strength.

## Uniqueness Threats

1. Reconstruction is not communication. USAE, SAE universality, model
   stitching, and target-aligned PCA can reconstruct coordinates without
   improving target answers.
2. Atom identity can be arbitrary. Any positive row must fail atom-ID shuffle,
   coefficient shuffle, magnitude-only, atom-only, and top-atom knockout in the
   expected direction.
3. Exclusive/private features may be artifacts. Standard crosscoder L1 losses
   can misattribute shared concepts as exclusive; use BatchTopK/Latent Scaling
   style diagnostics.
4. SAE atoms are not canonical. Seed repeats, meta-feature checks, and
   stability/Jaccard telemetry are required before interpreting atom IDs.
5. Target-cache shortcuts remain the main confound. Target-derived soft slots,
   slots-only, zero-source, and target-family substitution must stay in the
   first gate, even on n8/n32.
6. Behavior-diffing is not answer repair. Crosscoder/transcoder papers isolate
   behaviors inside or between models; LatentWire must show packet atoms cause
   downstream target answer changes under source-private constraints.
7. Cross-family novelty is fragile. Same-family shared atoms are insufficient
   for ICLR unless at least one strict cross-family pair is nonnegative and
   beats substitution controls.

## Mac-Feasible Experiments

1. Target behavior-transcoder atom scout.
   - Surface: ARC n8/n32 disagreement slice, then HellaSwag 128 only if n32
     clears controls.
   - Build: use Qwen-3 pretrained transcoders if model-compatible; otherwise
     train a tiny single-layer transcoder on cached target candidate residuals
     with a target-margin auxiliary head.
   - Packet: top-k feature IDs plus signed quantized activations per candidate.
   - Decode: add a train-only feature-to-target-margin decoder, not a
     soft-prefix-only receiver.
   - Controls: atom shuffle, coefficient shuffle, top-atom knockout,
     source-row shuffle, target-derived, same-byte text/random,
     source-index/rank/score, candidate-roll, source-family substitution.

2. Shared/private crosscoder packet scout.
   - Surface: same ARC n8/n32 cache.
   - Build: train a small BatchTopK crosscoder over paired source/target
     candidate hidden innovations with explicit shared atoms and private atoms.
   - Packet: shared atom ID, source-private residual atom ID, two quantized
     coefficients, calibrated source uncertainty header.
   - Decode: target score plus learned atom-margin correction.
   - Telemetry: shared/private norm ratio, decoder cosine, latent scaling,
     active-code Jaccard, dead-code rate, atom knockout delta, packet bytes.

3. Delta-crosscoder source-only repair bucket.
   - Surface: examples where source is right and target is wrong or target
     top2 contains the right answer.
   - Build: train atoms to discriminate repair-positive vs repair-negative
     paired activations, not to minimize global reconstruction.
   - Gate: improvement must concentrate in source-only-correct buckets without
     increasing target-only-correct harm.

4. Pretrained Qwen atom feasibility probe.
   - Surface: target-side only first, no source packet claim.
   - Build: run Qwen-Scope or Qwen-3 transcoder feature extraction on candidate
     prompts; rank atoms by target answer-margin attribution/ablation.
   - Gate: if target-side atoms cannot causally move target margins on n32,
     do not spend a cycle training cross-model atoms.

5. Transferred-SAE cheap initializer.
   - Surface: source TinyLlama/Qwen or Qwen/Phi cached residuals.
   - Build: affine-map SAE or transcoder weights from a small compatible model
     into the target residual space, then fine-tune only the feature-to-margin
     decoder.
   - Gate: must beat freshly trained PCA and random feature dictionaries under
     the same byte cap.

## Decision

Promote behavior-oriented transcoders and BatchTopK shared/private crosscoders
as the top next branch. Demote further source-only or target-aligned PCA/common
coordinate sweeps unless they are only used as controls. The first gate should
be a Mac-local n32 behavior-atom scout with causal atom knockout and a
margin-decoder receiver; widening before that would repeat the failed
coordinate-alignment loop.
