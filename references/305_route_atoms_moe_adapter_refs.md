# Route Atoms, MoE Routing, and Adapter Gating

This memo collects recent routing-heavy papers that are directly relevant to LatentWire's next branches: route atoms, MoE-style routing, token/head specialization, adapter/FiLM/LoRA gating, and latent bottleneck routing.

## Why these papers matter

LatentWire is now in the regime where the main question is not "can we move information?" but "what is the smallest, most stable routing interface that preserves useful structure?"

The recurring design patterns here are:

- route through a small latent set instead of directly through all tokens
- make routing content-aware but interpretable
- separate query selection from value transport when it helps
- keep specialization sparse enough to inspect, but not so sparse that routing collapses
- prefer a fixed geometry or basis when we want stable routes
- treat adapter choice as a routing problem, not just a fine-tuning trick

## Primary sources

- [FLARE: Fast Low-rank Attention Routing Engine](https://arxiv.org/abs/2508.12594)
  - Routes attention through a short latent sequence of learned query tokens.
  - Useful for thinking about LatentWire route atoms as a low-rank attention basis rather than a raw token subset.

- [Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning](https://arxiv.org/abs/2505.16950)
  - Rewrites KV cache segments at step boundaries using a cache processor.
  - Relevant for periodic memory consolidation, cache rewrites, and whether LatentWire should rewrite past state instead of only selecting it.

- [DTRNet: Dynamic Token Routing Network to Reduce Quadratic Costs in Transformers](https://arxiv.org/abs/2509.00925)
  - Tokens can skip quadratic mixing while still receiving lightweight updates.
  - Good reference for token-level compute allocation and the question of which tokens need full transport.

- [Improving Routing in Sparse Mixture of Experts with Graph of Tokens](https://arxiv.org/abs/2505.00792)
  - Makes routing depend on token interactions rather than independent token decisions.
  - Useful if LatentWire needs pairwise or neighborhood-aware routing scores instead of per-token scores only.

- [SHRP: Specialized Head Routing and Pruning for Efficient Encoder Compression](https://arxiv.org/abs/2512.20635)
  - Treats attention heads as independent experts and uses usage-driven routing/pruning.
  - Useful for head-specialized route atoms and for deciding whether some heads should always stay dense.

- [ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](https://arxiv.org/abs/2511.10971)
  - Reparameterizes experts in an orthonormal basis and routes by cosine similarity to the basis.
  - Strong inspiration for route atoms that live in a normalized basis with an explicit geometry knob.

- [Grassmannian Mixture-of-Experts: Concentration-Controlled Routing on Subspace Manifolds](https://arxiv.org/abs/2602.17798)
  - Makes routing entropy a continuous function of a concentration parameter.
  - Useful if we want an explicit sparsity knob for route atoms, not just top-k.

- [MoLoRA: Composable Specialization via Per-Token Adapter Routing](https://arxiv.org/abs/2603.15965)
  - Routes individual tokens to adapters using either vocabulary structure or learned gating.
  - Directly relevant to adapter routing, multimodal token type splits, and per-token specialization.

- [AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization](https://arxiv.org/abs/2603.11873)
  - Pre-gates adapters once per token, then applies the same routing everywhere.
  - Good reference for "decide once, apply everywhere" gating and for deployment-friendly routing.

- [Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](https://arxiv.org/abs/2506.11672)
  - Dynamically allocates LoRA experts across layers and routes instructions layer-wise.
  - Useful for layer-wise adapter budgets and for measuring whether early vs late layers want different route atoms.

- [Unifying Mixture of Experts and Multi-Head Latent Attention for Efficient Language Models](https://arxiv.org/abs/2508.01261)
  - Combines MoE with latent attention and shared experts.
  - Strong source for multi-expert plus latent-memory hybrids and for separating common vs specialized compute.

- [From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing](https://arxiv.org/abs/2510.03293)
  - Re-routes based only on the score distribution shape and load balance.
  - Useful if LatentWire needs an inference-time routing fix without retraining.

- [Adaptive gating via SAGA: Selective Adaptive Gating for Efficient and Expressive Linear Attention](https://arxiv.org/abs/2509.12817)
  - Adapts KV aggregation with input-dependent gates to reduce low-rank collapse.
  - Good template for selective gating over translated K/V rather than uniform compression.

- [TransNAR: Transformers meet Neural Algorithmic Reasoners](https://arxiv.org/abs/2406.09308)
  - Uses gated cross-attention to inject a second structured signal into a Transformer.
  - Relevant if LatentWire adopts an auxiliary structured channel or a two-stream bridge.

- [COLT5: Faster Long-Range Transformers with Conditional Computation](https://arxiv.org/abs/2303.09752)
  - Classic but still useful: routed query tokens, routed KV tokens, and separate attention and FFN budgets.
  - Good for exactly how to split query selection from key/value selection.

- [Semantic Resonance Architecture: Interpretable LLMs via Semantic Anchors](https://arxiv.org/abs/2509.14255)
  - Uses cosine routing to semantic anchors and an orthogonality/dispersion loss.
  - Relevant for interpretable route atoms and semantic anchor telemetry.

- [ZeroRouter: Cost-Efficient Zero-Shot LLM Routing via a Universal Latent Space](https://arxiv.org/abs/2601.06220)
  - Routes through a model-agnostic latent space.
  - Useful as a routing abstraction if LatentWire needs a model-agnostic bridge space.

## Useful math ideas to steal

- Low-rank route factorization:
  - `h_out = sum_i g_i a_i`
  - `g = softmax(W_r h)` or `g = topk(softmax(W_r h))`
  - `a_i` can be route atoms, adapter atoms, or head atoms.

- Geometry-aware routing:
  - route by cosine similarity to normalized atoms or basis vectors.
  - constrain atoms with orthogonality or dispersion regularization.
  - track the singular spectrum of the atom bank; flat spectra are usually a sign of collapse resistance.

- Separate query and memory budgets:
  - choose query atoms with one router, then choose transported KV atoms with another router.
  - this is the COLT5/FLARE split, but in LatentWire terms.

- Continuous sparsity instead of hard sparsity:
  - replace hard top-k with a temperature or concentration parameter.
  - gives an interpretable knob for route entropy and dead-atom rate.

- Residual modulation vs multiplicative modulation:
  - additive residual adapters preserve base behavior better.
  - multiplicative FiLM-style gates are useful when a dimension-wise scaling signal is the right inductive bias.

## Eight concrete LatentWire ablations

1. Route-atom budget sweep
   - Vary atom count per layer/head: `1, 2, 4, 8, 16`.
   - Telemetry: task accuracy, paired delta vs target-alone, route entropy, dead-atom rate, atom usage Gini, latency, KV bytes.

2. Shared vs per-layer route atoms
   - Use one atom bank across all layers versus a separate bank per layer.
   - Telemetry: layer localization, atom reuse rate, cross-layer cosine overlap, accuracy.

3. Hard top-k vs soft routing
   - Compare hard top-k selection, softmax mixture, and temperature-controlled top-k.
   - Telemetry: entropy, top-1 margin, load balance, calibration, collapse rate, robustness under noise.

4. Query-routing vs query-plus-KV routing
   - Route query tokens only, route KV only, and route both separately.
   - Telemetry: query/KV route disagreement, selected span fractions, prefix retention, accuracy delta.

5. Adapter modulation family
   - Compare additive residual adapters, LoRA injection, and FiLM-style scale-shift gating.
   - Telemetry: norm drift, channel sparsity, parameter count, accuracy per budget.

6. Head specialization ablation
   - Tie route atoms across heads versus make them head-specific.
   - Telemetry: per-head specialization entropy, head usage histograms, head collapse rate, attention-head mutual information.

7. Latent bottleneck width and rewrite frequency
   - Sweep bottleneck width and periodic rewrite interval.
   - Telemetry: bottleneck occupancy, information retention proxy, recency bias, prefix reconstruction error, generation stability.

8. Quantization-inspired bridge preconditioning
   - Add orthogonal rotation, per-channel scaling, or low-rank preconditioning before routing.
   - Telemetry: condition number, singular values, cosine preservation, route stability under perturbations, accuracy.

## Telemetry to keep in every run

- route entropy per layer and per head
- dead-atom / dead-head rate
- top-1 margin and score distribution shape
- atom usage Gini / entropy
- layer-localization of selected atoms
- query/KV route disagreement
- prefix retention and reconstruction error
- K/V norm drift after transport
- singular spectrum of atom banks
- paired delta versus target-alone and versus dense-aligned baseline
- latency, KV bytes, and active compute

## Recommended interpretation

Treat route atoms as the bridge analog of MoE experts:

- experts become atoms
- routing logits become query-to-atom similarity
- specialization becomes a geometry problem
- the key risk is collapse, not expressivity

So the right failure modes to watch are:

- too few atoms dominate
- atoms become duplicated and non-orthogonal
- query and KV routing drift apart
- performance improves only by adding compute, not by improving selection

The memo should be used as a design checklist for the next LatentWire branches, not as evidence that the current bridge is already solved.
