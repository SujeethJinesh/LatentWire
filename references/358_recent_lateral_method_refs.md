# Recent Lateral Method References for LatentWire

Web check: 2026-04-21. Primary-source memo for recent methods that could inspire cross-model latent communication, with an emphasis on projector design, routing, iterative refinement, cache/quantization controls, tokenizer adaptation, and representation alignment.

## 1) Projectors, adapters, and routed interfaces

- **[DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models](https://openreview.net/forum?id=Rx5LhMMr0c)**, **[Spatial-Aware Efficient Projector for MLLMs via Multi-Layer Feature Aggregation](https://arxiv.org/abs/2410.10319)**, and **[Dynamic Multi-Expert Projectors with Stabilized Routing for Multilingual Speech Recognition](https://arxiv.org/abs/2601.19451)**. Relevance: these three papers all argue that the interface should preserve structure while avoiding a single monolithic compressor; for LatentWire, that suggests testing spatial-aware or task-routed bridges instead of only widening a dense projector. Ablations: single dense bridge vs spatial projector vs routed experts; routed experts with soft vs sparse gating; layer-sparse vs all-layer bridge injection; log routing entropy, expert load balance, and accuracy at matched bytes.

- **[MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models](https://arxiv.org/abs/2506.05928)** and **[Activated LoRA: Fine-tuned LLMs for Intrinsics](https://research.ibm.com/publications/activated-lora-fine-tuned-llms-for-intrinsics)**. Relevance: heterogeneous adapter mixtures and invocation-aware adapters are strong evidence that the bridge can be modular and context-triggered rather than always-on; that is directly relevant if LatentWire needs multiple latent "skills" for different source-target regimes. Ablations: one bridge vs heterogeneous adapter mixture vs invocation-gated adapter; soft vs sparse expert selection; cache reuse with and without adapter activation boundaries.

## 2) Diffusion-style latent refinement

- **[Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens](https://arxiv.org/abs/2504.14666)**, **[Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)**, and **[LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://openreview.net/forum?id=z5cPEZ4n6i)**. Relevance: these methods all point to iterative denoising as a better interface than one-shot decoding when the latent state is underdetermined; for LatentWire, the natural hypothesis is that a bridge may work better as a short refinement chain than as a single projection. Ablations: one-pass bridge vs k-step denoiser vs blockwise diffusion; discrete stage tokens vs none; parallel sample count vs compute-normalized accuracy; log convergence rate, rollback count, and answer diversity.

## 3) KV cache compression and selection

- **[Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://openreview.net/forum?id=KzACYw0MTV)**, **[Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution](https://arxiv.org/abs/2510.00636)**, and **[KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](https://arxiv.org/abs/2505.23416)**. Relevance: these are the strongest recent controls for "compressed memory" under long context, and they separate current-query sparsity from future-query utility and from reconstruction-based reuse. Ablations: query-aware vs future-query expected attention vs query-agnostic reconstruction; same-model cache curation vs cross-model transport; log retained bytes, latency, retrieval accuracy, and sensitivity by prompt family.

## 4) Quantization geometry

- **[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://proceedings.mlr.press/v235/liu24bz.html)** and **[KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://openreview.net/forum?id=0LXotew9Du)**. Relevance: both show that key and value states want different treatment, and that pre-RoPE placement plus asymmetric statistics matter; LatentWire should treat its latent payload the same way instead of forcing a symmetric bottleneck. Ablations: symmetric vs asymmetric bits; per-channel vs per-token quantization; pre-RoPE vs post-RoPE key handling; with and without outlier buckets.

- **[QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)** and **[Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118)**. Relevance: these are the best reminders that the geometry of the codebook matters as much as the bit budget; if LatentWire can move latent messages through a small codebook, codebook structure may beat a plain linear map. Ablations: linear bridge vs Hadamard/lattice codebook vs additive codebook; codebook depth vs residual norm; dead-code rate and utilization entropy at matched bitrate.

## 5) Tokenizer and vocabulary adaptation

- **[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)**, **[Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083)**, and **[Model-Aware Tokenizer Transfer](https://openreview.net/forum?id=IyV1QEc95F)**. Relevance: these papers make the same point from three angles: tokenizer mismatch is not just a nuisance, it can be the bottleneck that hides bridge quality. Ablations: bridge-only vs TokAlign-style vocab remap vs cross-tokenizer distillation warm-up; byte-level fallback vs subword transfer; log remap purity, perplexity recovery, and downstream transfer by tokenizer gap.

## 6) Representation alignment and transport

- **[Transport and Merge: Cross-Architecture Merging for Large Language Models](https://arxiv.org/abs/2602.05495)** and **[Probabilistic Geometric Alignment via Bayesian Latent Transport for Domain-Adaptive Foundation Models](https://arxiv.org/abs/2603.23783)**. Relevance: OT-style cross-neuron matching and probabilistic latent transport are the most direct recent references for initializing a bridge when source and target representations are genuinely heterogeneous; they are especially relevant for mismatched model families. Ablations: direct bridge training vs OT-initialized bridge vs probabilistic transport regularizer; deterministic vs uncertainty-aware alignment; log alignment residuals, calibration, and downstream accuracy.

## Priority ablations

1. **Dense bridge vs routed bridge.** Compare a single projector against spatial-aware and MoE-style projectors on one homogeneous pair and one heterogeneous pair.
2. **Query-aware vs query-agnostic memory control.** Run Quest, Expected Attention, and KVzip-style controls under the same byte budget before claiming any communication win.
3. **Tokenizer first, transport second.** Test TokAlign or model-aware tokenizer transfer before bridge training and measure whether the latent interface still needs heavy alignment machinery.
4. **Asymmetric latent quantization.** Sweep key/value bit splits and codebook-based latent packing to see whether one side of the bridge is the true bottleneck.
5. **OT initialization and uncertainty regularization.** For the hardest model pair, initialize with transport alignment and then add a probabilistic latent regularizer; keep one deterministic control for calibration.
