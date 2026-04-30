# Contrastive Receiver Pruning References

- date: `2026-04-30`
- blocker: the paper needs a less hand-shaped learned receiver than the explicit semantic-anchor overlap decoder, but it must preserve strict source-destroying controls.
- role: primary-source memo for the learned contrastive receiver probe and pruning decision.

## Sources And Experiment Impact

1. **Sparse Crosscoders for Cross-Layer Features and Model Diffing**
   - Source: https://transformer-circuits.pub/2024/crosscoders/index.html
   - Blocker helped: makes the shared/private atom interface and causal feature knockout legible to reviewers.
   - Mechanism/design idea: represent communication through sparse shared features plus source/model-specific residuals, then test causal knockout.
   - Experiment change: keep top-feature knockout and shared/private atom accounting as required diagnostics; do not present a learned receiver without a feature-causality ablation.
   - Role: inspiration, interpretability support, and related work.

2. **Cross-Architecture Model Diffing with Crosscoders**
   - Source: https://arxiv.org/abs/2602.11729
   - Blocker helped: cross-family generalization remains the central novelty risk.
   - Mechanism/design idea: learned dictionaries should separate shared features from architecture-specific features.
   - Experiment change: require bidirectional core/holdout passes before promoting any learned receiver; same-family positives are insufficient.
   - Role: closest related learned-dictionary baseline and novelty threat.

3. **Universal Sparse Autoencoders**
   - Source: https://arxiv.org/abs/2502.03714
   - Blocker helped: the paper needs a credible path from hand anchors to model-spanning learned dictionaries.
   - Mechanism/design idea: use one shared sparse dictionary over multiple model or surface distributions and report shared support/reconstruction diagnostics.
   - Experiment change: future receiver branches should report support overlap and reconstruction/headroom, but promotion still depends on task lift under controls.
   - Role: baseline and method inspiration.

4. **Group Crosscoders for Mechanistic Analysis of Symmetry**
   - Source: https://arxiv.org/abs/2410.24184
   - Blocker helped: synonym and paraphrase robustness need an invariance principle instead of hand synonym lists.
   - Mechanism/design idea: train over transformation orbits to expose invariant and equivariant features.
   - Experiment change: the held-out paraphrase split remains mandatory; exact transformed-surface overlap must stay zero.
   - Role: inspiration and ablation design.

5. **SimCSE**
   - Source: https://aclanthology.org/2021.emnlp-main.552/
   - Blocker helped: public candidate text must align paraphrases while separating wrong candidates.
   - Mechanism/design idea: contrastive positives and in-batch negatives can learn semantic compatibility.
   - Experiment change: the current contrastive receiver probe was the direct test of this idea; it failed strict cross-family promotion.
   - Role: semantic baseline and ablation design.

6. **Wyner-Ziv Source Coding**
   - Source: https://ieeexplore.ieee.org/document/1055508/
   - Blocker helped: reviewers need a precise communication-theory framing for tiny source-private packets decoded with target side information.
   - Mechanism/design idea: send a compressed source residual when the decoder has correlated side information.
   - Experiment change: keep framing the method as side-information decoding, not protocol-free latent transfer.
   - Role: theory support and paper framing.

7. **Optimized Product Quantization**
   - Source: https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
   - Blocker helped: separates codec primitives from the paper's actual contribution.
   - Mechanism/design idea: compact codebook indices are a baseline transport primitive.
   - Experiment change: keep PQ/OPQ as compression baselines; do not claim novelty for product-codebook encoding itself.
   - Role: baseline and systems ablation.

8. **Cache-to-Cache**
   - Source: https://arxiv.org/abs/2510.03215
   - Blocker helped: broad cross-model communication has a high-rate learned-cache competitor.
   - Mechanism/design idea: learned KV/cache transfer can improve target decoding when source and target states are available.
   - Experiment change: position LatentWire as an extreme-rate source-private packet interface with systems/privacy tradeoffs, not as a direct KV-transfer replacement until native serving comparisons exist.
   - Role: competitor and novelty boundary.

## Decision

The learned contrastive receiver remains an ablation, not a headline method.
On the held-out paraphrase split, the unconstrained bilinear receiver keeps
matched source signal but leaks under atom-ID derangement; adding source-control
negatives fixes controls but suppresses core-to-holdout matched lift. The next
method branch needs a new mechanism, not more threshold tuning of this exact
bilinear receiver.
