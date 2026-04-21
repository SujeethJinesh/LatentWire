# Route Atoms and Latent Communication References

Context:
- Current LatentWire evidence says static bridges, local prediction teachers, and output-row preference distillation can recover small-slice behavior but do not yet beat target-alone on controlled GSM slices.
- The next positive-method path should test whether cross-model communication needs a more explicit interface: route atoms, head-wise routing, shared latent/byte bottlenecks, or adapter mixtures with interpretable capacity separation.
- This memo only records recent primary-source inspiration and concrete ablations.

## 1) Route atoms and routing geometry

- [Multilingual Routing in Mixture-of-Experts](https://arxiv.org/abs/2510.04694), 2025/2026 ICLR. The paper finds early/late routing is language-specific while middle-layer routing has cross-lingual alignment, and inference-time steering of shared middle-layer experts improves multilingual performance. For LatentWire, this argues for evaluating route atoms by layer and not averaging all layers into one transport policy.

- [Multi-Head Attention as a Source of Catastrophic Forgetting in MoE Transformers](https://arxiv.org/abs/2602.12587), 2026. This paper identifies a pre-routing bottleneck where concatenated multi-head attention states collapse separable factors before routing, then proposes head-wise routing to reduce composition collisions. LatentWire's current bridge likely has the same failure mode: query-conditioned banks may be too coarse if route decisions are made after head signals are already mixed.

- [Unifying Mixture of Experts and Multi-Head Latent Attention for Efficient Language Models](https://arxiv.org/abs/2508.01261), 2025. MoE-MLA-RoPE combines fine-grained micro-expert top-k routing with latent attention and shared always-on experts, reporting large KV-cache compression with small quality loss. This is directly relevant to a route-atom bridge: always-on shared atoms plus top-k specialized atoms may be more stable than a purely soft bank.

- [Scalable Prompt Routing via Fine-Grained Latent Task Discovery](https://arxiv.org/abs/2603.19415), 2026. The method discovers latent task clusters, then uses task-specific MoE heads for quality prediction across model pools. LatentWire can adapt the idea internally: cluster source-query/activation trajectories into route atoms before fitting target-side transport.

Concrete ablations:
- `route_atom_topk`: replace scalar bridge gates with 2/4/8 route atoms, selecting top-1 or top-2 atoms per layer/head from query features.
- `route_atom_shared_plus_specialist`: force one always-on shared atom plus one routed specialist atom; compare with unconstrained softmax mixtures.
- `headwise_route_atom`: learn route coefficients per attention head before concatenation; log route entropy, atom usage, and head-level collision counts.
- `middle_layer_only_route_steer`: fit route atoms only on layers with strongest source-target CKA/CCA overlap; compare to all-layer routing.

## 2) Shared bottlenecks for cross-model communication

- [Direct Semantic Communication Between Large Language Models via Vector Translation](https://arxiv.org/abs/2511.03945), 2025. The authors train learned vector translations between Llama-2-7B and Mistral-7B-Instruct and inject translated vectors at conservative blending strength, showing steerability without logit destabilization. This is the closest direct comparator to LatentWire and suggests our telemetry should separate alignment quality from generation stability under injection strength.

- [TransMLA: Multi-Head Latent Attention Is All You Need](https://arxiv.org/abs/2502.07864), 2025. TransMLA converts GQA models to MLA by introducing compressed latent KV states, preserving quality after moderate fine-tuning while reducing KV cache cost. LatentWire should test whether a shared latent KV bottleneck is easier to translate than full per-head K/V tensors.

- [Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution](https://arxiv.org/abs/2510.00636), 2025. Expected Attention estimates future-query importance for KV pruning without materialized future attention, and ships KVPress as a benchmark library. The LatentWire connection is to rank transported KVs by predicted target future utility, not by source-side attention alone.

- [Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions](https://arxiv.org/abs/2602.09483), 2026. Align-TI distills token interactions rather than only next-token distributions, including transition-style response dynamics. This supports an interaction-level teacher for LatentWire, but our failed local interaction variants imply it should be applied to route/transition structure, not raw prompt-local KL alone.

Concrete ablations:
- `latent_kv_bottleneck`: compress source K/V into a small latent basis, translate the latent, then reconstruct target K/V; compare against direct full-K/V bridge at matched byte budget.
- `expected_attention_transport`: weight calibration examples and transported positions by estimated future-query target utility; report whether gains come from better pruning or better alignment.
- `interaction_route_teacher`: distill teacher token-transition or attention-transition structure into route atom assignments rather than into output logits.
- `blend_sweep_stability`: run fixed injection strengths 0.05/0.10/0.20/0.30 and log accuracy, logit entropy shift, repetition rate, and answer-flip direction.

## 3) Tokenizer and hidden-space alignment

- [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083), 2025 NeurIPS. Approximate likelihood matching enables distillation across fundamentally different tokenizers and includes tokenizer transfer plus cross-tokenizer ensembling. For LatentWire, this is a strong argument to stop treating vocab mismatch as incidental; a shared tokenizer/objective may be needed before latent KV transport can be reliable.

- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466), 2026. Byte-Level Distillation maps teacher output distributions to byte probabilities and attaches a lightweight byte decoder to the student, using bytes as a common interface across tokenizers. LatentWire should add a byte-level auxiliary probe so transported latent states are evaluated through a tokenizer-independent channel.

- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523), 2025 ACL. TokAlign learns one-to-one vocabulary mappings from token co-occurrences, rearranges embeddings, and uses progressive fine-tuning to restore performance; after vocabulary unification, token-level distillation improves over sentence-level distillation. This suggests a practical control ablation: align vocabularies or byte-probe first, then test whether latent transport improves.

- [KL-based self-distillation for large language models](https://hf.co/papers/2508.15807), 2025. The paper targets vocabulary expansion in frozen LLMs through KL-based distillation, making distributional token knowledge a vocabulary-adaptation signal. Its relevance is narrower than ALM/BLD/TokAlign, but it supports logging token-family KL and not just exact-token accuracy.

Concrete ablations:
- `byte_probe_bridge`: train a frozen-target byte decoder/probe on target hidden states, then score transported hidden states through byte likelihood instead of target-token ID likelihood.
- `tokalign_control`: create source-target token co-occurrence mapping for calibration text; compare latent transport with and without mapped-token prompt canonicalization.
- `alm_span_teacher_v2`: retry span likelihood only after byte/token alignment, not on raw mismatched vocabularies.
- `token_family_metrics`: log digit/operator/unit/name byte-family accuracy, not only final GSM answer correctness.

## 4) Adapter mixtures and communication bottlenecks

- [CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters](https://arxiv.org/abs/2601.04885), 2026. CuMA frames conflicting alignment targets as conditional capacity separation and routes to specialized adapter subspaces to avoid mean collapse. LatentWire's negative averages may be analogous mean collapse across reasoning modes, so adapter mixtures should be route-conditioned rather than globally averaged.

- [Fine-Grained VLM Fine-tuning via Latent Hierarchical Adapter Learning](https://arxiv.org/abs/2508.11176), 2025. LatHAdapter uses learnable attribute prompts and hierarchy regularization to bridge category/text and image representations. For LatentWire, learnable latent attributes could become route atoms with hierarchical regularization over problem types or token roles.

- [RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models](https://arxiv.org/abs/2512.06811), 2025. RMAdapter adds a reconstruction branch that preserves original latent features while an adaptation branch injects task knowledge. LatentWire should use reconstruction loss as an anti-drift constraint for transported K/V, since many bridge variants appear to perturb without improving target reasoning.

- [Small Models, Big Impact: Efficient Corpus and Graph-Based Adaptation of Small Multilingual Language Models for Low-Resource Languages](https://hf.co/papers/2502.10140), 2025. This work reports adapter-based low-resource adaptation with embedding alignment and bottleneck/LoRA variants. It is less directly about cross-model communication, but it motivates cheap adapter controls against full bridge retraining.

Concrete ablations:
- `reconstructive_bridge_adapter`: train a small adapter with two losses: target utility teacher plus reconstruction back to original target hidden/KV geometry.
- `latent_attribute_atoms`: initialize route atoms from clusters over token roles such as digits, operators, entities, and reasoning markers; compare to random route atoms.
- `adapter_moe_bridge`: route among LoRA/bottleneck adapters per layer/head instead of routing only bridge matrices.
- `mean_collapse_probe`: measure whether bridge outputs collapse route/token families by tracking per-family variance and nearest-centroid confusion.

## 5) Interpretation metrics to add before larger runs

- Route usage: per-layer/head atom histogram, entropy, top-k margin, and dead-atom count.
- Geometry: source-target CKA/CCA by layer, Procrustes residual, per-head cosine, and post-bridge norm drift.
- Bottleneck health: latent reconstruction error, byte-probe likelihood, token-family KL, and future-attention-weighted KV utility.
- Behavioral flips: target-alone vs bridge paired answer flips, logit entropy shift, refusal/repetition rate, and answer extraction failure rate.
- Capacity separation: variance retained per route family and mean-collapse score across prompt clusters.

## Priority stack

1. Implement `headwise_route_atom` first because it tests a concrete failure mode from 2026 MoE routing work: route collisions after head concatenation.
2. Add `byte_probe_bridge` as a tokenizer-independent diagnostic before retrying likelihood teachers; ALM/BLD/TokAlign all imply raw token mismatch is a first-order blocker.
3. Run `latent_kv_bottleneck` against direct K/V transport at matched byte budget; if latent bottlenecks outperform direct bridges, the positive paper angle becomes efficient shared communication rather than exact hidden-state translation.
