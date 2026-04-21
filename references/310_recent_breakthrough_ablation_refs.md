# 310 Recent Breakthroughs for LatentWire Ablations

Date: 2026-04-21

Scope: 2025-2026 primary sources relevant to cross-model latent communication, with emphasis on ablations we can run and telemetry that will make failures interpretable.

## Readout

The strongest next direction is not another single adapter variant. The recent literature points to a stacked method: symmetry-aware latent alignment, asymmetric K/V transport, token/codebook bridging, and optional iterative refinement. This matches our current toy positives: codebook remapping and asymmetric K/V budgets are plausible components, but they need to be lifted into evaluator runs with telemetry that can tell whether we improved routing, value reconstruction, tokenizer compatibility, or only reduced bytes.

## Primary Sources And What To Steal

### 1. Latent Attention And Hybrid Compute

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) introduces unified thinking/non-thinking operation, dense and MoE models, and explicit thinking budgets. For LatentWire, this argues for reporting an accuracy/latency/bytes frontier rather than one fixed reasoning setting.
- [Latent Multi-Head Attention for Small Language Models](https://arxiv.org/abs/2506.09342) studies MLA-style attention in small models. This is directly relevant because our bridge experiments are also in the small/open-model regime where MLA tradeoffs may differ from frontier-scale models.
- [EG-MLA](https://arxiv.org/abs/2509.16686) adds embedding-gated MLA to reduce KV cache while increasing expressiveness. This suggests gating the bridge by token/position embeddings rather than only using fixed transport ratios.
- [TransMLA / Enabling MLA in existing Transformers](https://aclanthology.org/2025.acl-long.1597/) is important because it retrofits MLA into existing models. This is conceptually close to LatentWire: retrofit a latent communication interface without full pretraining.
- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804) uses Hybrid Attention MoE in both Thinker and Talker with long context. This reinforces the idea that the communicating submodules can have different attention/compression policies.

Actionable ablations:

- `mlp_adapter` vs `orthogonal_adapter` vs `embedding_gated_adapter`, where the gate is conditioned on source token embedding norm, target token embedding norm, and attention entropy.
- Layerwise K/V transport sweep with separate route/value budgets and an early-layer precision boost, then compare to uniform budgets.
- "Think budget" sweep: fixed latent bridge steps vs adaptive bridge steps selected by entropy or verifier confidence.

Telemetry to add:

- Per-layer active K/V counts, route/value overlap, route/value Jaccard, attention KL before/after bridge, and per-layer negative-transfer flags when adding more bridge bandwidth reduces accuracy.

### 2. Multimodal Latent Interfaces

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) uses dynamic resolution processing, window attention, and temporal encoding for video. The relevant pattern is not vision itself; it is variable-resolution latent ingestion.
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) uses block-wise streaming encoders, TMRoPE, and a Thinker-Talker architecture where hidden representations from one module condition another modality decoder.
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765) extends the Thinker-Talker MoE pattern across text, image, audio, and video.
- [Multimodal Reasoning via Latent Refocusing](https://arxiv.org/abs/2511.02360) combines latent reasoning with visual refocusing and joint alignment/reconstruction losses. The useful idea is an iterative "refocus" readback into the source evidence, not just a one-shot projection.

Actionable ablations:

- Treat a source model's hidden states as a "foreign modality": encode to a small latent token set, decode into target prefix/KV, and add a source reconstruction loss or diagnostic even when the final task is text-only.
- Add refocus passes: bridge once, score target uncertainty, then select another small source-token subset and update the bridge.
- Compare fixed prefix latents to query-conditioned latents, where target hidden states query source hidden states like a multimodal connector.

Telemetry to add:

- Source-token coverage by refocus pass, marginal accuracy gain per pass, reconstruction cosine by layer, target uncertainty reduction, and source-target attention concentration after each refocus.

### 3. Token, Vocab, And Codebook Alignment

- [TokAlign](https://aclanthology.org/2025.acl-long.207/) aligns source and target vocabularies through token co-occurrences and progressively fine-tunes after rearranging embeddings. This maps directly to our cross-tokenizer bridge problem.
- [AdaptiVocab](https://arxiv.org/abs/2503.19693) adapts vocabularies to reduce domain token usage by replacing tokens with domain n-grams and initializing embeddings from existing embeddings.
- [FLEXITOKENS](https://arxiv.org/abs/2507.12720) learns flexible byte boundaries to reduce over-fragmentation during adaptation.
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) is slightly outside the date window by submission date, but it is central background for 2025 tokenization work: entropy-based byte patches allocate compute where tokenization is uncertain.
- [Token Assorted](https://arxiv.org/abs/2502.03275) mixes text tokens with VQ-VAE latent tokens to compress reasoning traces.

Actionable ablations:

- Extend `codebook_remap` from toy into a real bridge: learn a small shared codebook over source hidden states, then map code IDs to target-side pseudo-tokens or prefix vectors.
- Token-overlap stratification: report bridge success separately for prompts with high vs low tokenizer overlap between source and target.
- Co-occurrence bridge: initialize latent code assignment from token co-occurrence or byte-span alignment, then compare to random and Procrustes initializations.
- Entropy patching: allocate more latent bridge slots to high tokenizer-disagreement spans, high source entropy spans, or high target surprise spans.

Telemetry to add:

- Tokenizer overlap, byte-span overlap, compression ratio by prompt, codebook entropy, dead code rate, code collision, source-token-to-code mutual information, target-token-to-code mutual information, and accuracy stratified by tokenizer mismatch.

### 4. Diffusion And Iterative Latent Refinement

- [Large Language Diffusion Models / LLaDA](https://arxiv.org/abs/2502.09992) shows masked diffusion language modeling can be competitive with autoregressive LMs.
- [Block Diffusion](https://arxiv.org/abs/2503.09573) interpolates between autoregressive and diffusion LMs and keeps KV caching while sampling blocks in parallel.
- [Dream 7B](https://arxiv.org/abs/2508.15487) is an open diffusion LLM useful as a practical reference for instruction-tuned diffusion generation.
- [LaDiR](https://arxiv.org/abs/2510.04573) uses latent diffusion over reasoning blocks with bidirectional block attention and adaptive test-time compute.
- [System-1.5 Reasoning](https://arxiv.org/abs/2505.18962) uses latent shortcuts and self-distillation to skip non-critical reasoning steps.

Actionable ablations:

- Iterative denoising bridge: corrupt or truncate the source latent message, then run 1, 2, 4, and 8 refinement passes before target decoding.
- Block bridge: communicate fixed-size latent blocks instead of single prefix vectors, and let later blocks condition on earlier target uncertainty.
- Early-exit latent bridge: skip low-entropy positions and spend bridge slots only on high-entropy or verifier-critical positions.

Telemetry to add:

- Accuracy vs refinement steps, entropy reduction per step, edit distance between latent messages across iterations, verifier confidence movement, and failure class after each iteration.

### 5. Quantization, Compression, And Math-Inspired Transport

- [SVDq](https://arxiv.org/abs/2502.15304) projects K cache into SVD latent channels and applies importance-aware mixed precision. This directly supports testing latent-channel ordering instead of raw-dimension transport.
- [TurboQuant](https://arxiv.org/abs/2504.19874) uses random rotations plus scalar quantizers to get near-optimal vector distortion and quality-neutral KV quantization at moderate bit rates. This is strong support for rotation/Hadamard preprocessing before codebook transport.
- [Titanus](https://arxiv.org/abs/2505.17787) combines on-the-fly KV pruning and quantization. This suggests testing joint prune+quantize policies rather than evaluating them independently.
- [SageAttention3](https://arxiv.org/abs/2505.11594) uses microscaling FP4 attention and explores low-bit training, relevant for quantized attention transport.
- [Qwen3 Quantization Study](https://arxiv.org/abs/2505.02214) shows model- and task-specific quantization sensitivity, which argues against treating compression as a uniform post-processing knob.
- [TurboAngle](https://arxiv.org/abs/2603.27467) uses Hadamard-domain angle quantization, independently configures K/V codebooks per layer, and reports K-dominated vs V-dominated bottleneck patterns.
- [InnerQ](https://arxiv.org/abs/2602.23200) is a 2026 tuning-free KV-cache quantization method that reports GSM8K-preserving low-bit cache compression. This is a direct prompt to add task-aware KV quantization controls rather than only testing latent-vector MSE.
- [Sequential KV Cache Compression via Probabilistic Language Tries](https://arxiv.org/abs/2604.15356) argues for sequence-level KV compression rather than independent per-vector compression. Even if theoretical, it suggests cross-example prefix reuse telemetry.
- [AWQ](https://arxiv.org/abs/2306.00978) and its official implementation ([mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)) are older but still useful: activation-aware protection of salient channels maps cleanly to protecting bridge dimensions that control target attention logits.
- [EXL2 / ExLlamaV2](https://github.com/turboderp-org/exllamav2) is an engineering reference for mixed-bit allocation and practical bit-rate sweeps. It is not a KV bridge competitor, but its "choose precision by measured sensitivity" workflow should inform our per-layer/per-head bridge budget search.

Actionable ablations:

- Rotation family sweep before bridge: identity, orthogonal Procrustes, random Hadamard, learned Householder stack, SVD basis, and whitening+rotation.
- Mixed precision bridge: allocate more bits/slots to high singular-value channels, high attention-gradient channels, or layers with measured route sensitivity.
- K/V asymmetry: independently quantize or select K and V, including "more K fewer V" and "fewer K more V" regimes.
- Prefix trie reuse: cache bridge states for repeated prompt prefixes and measure whether the latent bridge can reuse or delta-code source-prefix states.
- Activation-aware bridge budgeting: estimate per-channel or per-head saliency from target attention-logit movement, then protect a small high-saliency subset as AWQ-style full-precision residuals while compressing the rest.
- Mixed-bit latent transport: EXL2-style bit allocation over bridge channels or codebook residuals, reporting average bits-per-value rather than a single global quantization setting.

Telemetry to add:

- Quantization MSE, inner-product bias, attention-logit KL, singular spectrum retained, protected-channel fraction, average bits per K/V/cache/value residual, layerwise sensitivity curves, K-dominated/V-dominated layer labels, prefix reuse rate, and delta norm after predictive coding.

### 6. Symmetry, Gauge, And Model-Merging Insights

- [Beyond the Permutation Symmetry of Transformers](https://arxiv.org/abs/2502.00264) introduces continuous rotation symmetry for transformer parameter matching and improves model fusion. This is the closest direct mathematical reference for our observed rotation/alignment issues.

Actionable ablations:

- Gauge-aware bridge initialization: compare raw linear, orthogonal Procrustes, block-diagonal per-head Procrustes, rotation+scale, and head-permutation+rotation.
- Head-space matching: match source and target heads by attention-map similarity before learning the bridge.
- Symmetry stress test: apply random orthogonal rotations, head permutations, sign flips, and layer swaps in toy setups to see which bridge variants are invariant.

Telemetry to add:

- Orthogonality error, determinant/sign statistics, per-head assignment entropy, head-match stability across seeds, CKA/SVCCA before/after alignment, and performance under synthetic gauge transformations.

## Proposed Next Experiment Ladder

1. Run the real evaluator with the new asymmetric K/V selector across route/value ratios: `(0.25,0.75)`, `(0.50,0.50)`, `(0.75,0.25)`, and per-layer early boost if implemented.
2. Promote `codebook_remap` from toy to a bridge-side ablation with codebook sizes `16,32,64,128`, dead-code regularization, and tokenizer-overlap stratification.
3. Add a rotation-preconditioned codebook bridge: `Hadamard -> codebook -> target prefix/KV`, compared against Procrustes-only and codebook-only.
4. Add a two-pass refocus bridge: first bridge produces target uncertainty, second bridge spends extra source slots only on high-uncertainty spans.
5. Add a small diffusion-style latent refinement toy before touching full evaluator: denoise bridge messages under rotation, token permutation, and outlier channels.
6. Run C2C/KVComm/Quest/KVzip competitors on identical source-target model pairs and report accuracy vs bytes, not just raw accuracy.

## Minimum Paper-Safe Telemetry Schema

Every run should emit:

- `task`, `source_model`, `target_model`, `tokenizer_pair`, `prompt_id`, `seed`, `method`, `bytes_total`, `bytes_k`, `bytes_v`, `bytes_prefix`, `latency_prefill`, `latency_decode`, `accuracy`, `exact_match`, and `judge_score` where applicable.
- `tokenizer_overlap`, `byte_span_overlap`, `source_token_count`, `target_token_count`, `compression_ratio`, and `reasoning_budget`.
- `layer_k_keep_fraction`, `layer_v_keep_fraction`, `route_value_jaccard`, `attention_kl`, `value_recon_cosine`, `hidden_recon_cosine`, and `bridge_output_norm`.
- `codebook_size`, `codebook_entropy`, `dead_code_rate`, `collision_rate`, `assignment_margin`, and `source_target_code_mi` for discrete bridge methods.
- `rotation_type`, `orthogonality_error`, `sv_energy_retained`, `quant_bits_k`, `quant_bits_v`, `quant_mse`, and `inner_product_bias` for compression/alignment methods.
- `refinement_steps`, `entropy_reduction`, `uncertainty_before`, `uncertainty_after`, and `marginal_gain_per_step` for iterative methods.

## Current Decision

Additively adding ideas to the paper is useful only if they become controlled axes in the method, not scattered features. The highest-value stack to pursue next is:

1. Symmetry-aware rotation/preconditioning.
2. Asymmetric K/V route-value allocation.
3. Discrete codebook/token bridge for tokenizer mismatch.
4. Optional refocus or iterative refinement for hard examples.

This gives a coherent paper story: cross-model communication fails because latent spaces have gauge mismatch, tokenizer mismatch, and route/value asymmetry; LatentWire fixes these with gauge-aware alignment, token/codebook bridging, and budgeted K/V transport, then validates each fix with interpretable telemetry.
