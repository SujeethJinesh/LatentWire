# 307 Multimodal Resampler / Interface Notes

Primary sources to mine:
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
  Q-Former is the cleanest template for a learned latent interface: a small query bank that pulls only task-relevant information from a frozen encoder before handing it to an LLM.
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
  Perceiver Resampler plus gated cross-attention is the best reference for fixed-size latent compression with explicit cross-attn injection.
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
  Minimal projector baseline: a simple linear bridge from encoder features into the LLM can outperform heavier connectors when the interface is well-conditioned.
- [HyperLLaVA](https://arxiv.org/abs/2403.13447)
  Useful for dynamic projector tuning and expert-style modulation at the interface.
- [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)
  Best source for ablation discipline: connector choice, data mixture, and stage-wise training all matter.
- [BLIP-3 / Perceiver-resampler-style connector notes](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5/discussions/6)
  Not a paper link, but useful as a breadcrumb that BLIP-3 replaced Q-Former with a Flamingo-style resampler.

2025-2026 updates worth stealing:
- [TokenCarve](https://arxiv.org/abs/2503.10501)
  Information-preserving token compression. Strong template for “compress without collapsing semantics.”
- [LEO-MINI](https://arxiv.org/abs/2504.04653)
  Conditional token reduction plus mixture-of-experts. Good model for query-aware reduction instead of blind pruning.
- [Token-Shuffle](https://arxiv.org/abs/2504.17789)
  Spatial-token to channel-token reshaping. Relevant if LatentWire needs a geometry-preserving rearrangement before transport.
- [AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](https://arxiv.org/abs/2412.03248)
  Training-free token merging and layer pruning with importance estimates.
- [STAR: Stage-Wise Attention-Guided Token Reduction](https://arxiv.org/abs/2505.12359)
  Attention-guided pruning with a global view; useful for budgeted interface selection.
- [Top-Down Compression: Revisit Efficient Vision Token Projection for Visual Instruction Tuning](https://arxiv.org/abs/2505.11945)
  Good evidence that “compress from the top down” beats naive bottom-up token loss.
- [AdaptMerge](https://aclanthology.org/2025.findings-emnlp.387.pdf)
  Inference-time adaptive visual/language-guided token merging.
- [Delta-LLaVA](https://arxiv.org/abs/2512.18910)
  Base-then-specialize alignment; useful if the interface should first learn a generic bridge, then specialize.
- [TAMP: Token-Adaptive Layerwise Pruning in Multimodal Large Language Models](https://arxiv.org/abs/2504.09897)
  Layerwise sparsity driven by token diversity; good for head/layer budget coupling.

Math / interface ideas to steal for LatentWire:
- Use a small learned latent bank as the transport surface, not raw token passthrough.
- Treat queries as routing atoms: each atom pulls a different subspace of K/V, then a gate mixes them.
- Preserve geometry by keeping a fixed latent count and logging how much mass each latent absorbs.
- Separate “selection” from “projection”: first choose what survives, then map it into the shared interface.
- Add entropy or diversity regularization so the bridge does not collapse onto one atom/head.
- If a quantization-style preconditioner helps, interpret it as conditioning the interface before routing, not as a lossless compression trick.
- For cross-model comms, prefer budgeted transport with explicit keep-fraction telemetry over opaque end-to-end projection.

Concrete LatentWire ablations:
- Query/interface count sweep: `4 / 8 / 16 / 32`.
- Static queries vs learned queries vs query-conditioned queries.
- Linear projector vs MLP projector vs Q-Former-style cross-attn pooler.
- Attention-based token merge vs similarity-based merge vs top-down importance merge.
- Headwise route-atom selection vs dense attention vs budgeted pruning.
- Frozen interface vs lightly tuned interface vs fully trained interface.
- Preconditioned interface vs raw interface.
- Spatially aware merge vs token-only merge for any geometry-bearing inputs.
- Shared interface across source/target vs per-model interface specialization.

Telemetry to keep interpretable:
- Selected latent count, keep fraction, and per-layer/head survival rate.
- Attention entropy, score gap, and dead-atom rate.
- Overlap/Jaccard between selector prior and final selected tokens/heads.
- Norm ratios, cosine drift, and condition-proxy metrics for preconditioned interfaces.
- Latent occupancy histograms, collision rate, and per-example route traces.
- Accuracy paired with bytes / token count / TTFT / tokens-per-second so compression is not reported in isolation.
