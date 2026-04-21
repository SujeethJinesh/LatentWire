# Recent Latent / Multimodal Reasoning References for LatentWire

Web check: 2026-04-21. This memo prioritizes primary sources that can directly shape LatentWire ablations around latent/continuous reasoning, diffusion-style or iterative refinement, multimodal projector/resampler interfaces, and cross-model communication.

## 1) Latent / continuous reasoning

- **[Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)**. This is the clearest recent proof that an LLM can reason by feeding hidden states back as continuous thoughts instead of committing to every step in text. `Ablation:` replace raw source-token transport with a learned continuous-thought loop that feeds back a latent bridge state every `k` steps, then compare against the current tokenwise bridge.

- **[Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space](https://arxiv.org/abs/2505.15778)**. Soft Thinking uses probability-weighted mixtures of embeddings to keep reasoning in a smooth concept space while staying readable. `Ablation:` test a soft-token bridge where transported states are mixtures over target embeddings, and compare it to hard nearest-neighbor tokenization of the same latent summary.

- **[Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://openreview.net/forum?id=hYfOPXrbUr&noteId=uOxb1FtVLX)**. This mixes discrete latent tokens with text tokens so part of the reasoning trace can live in a compact latent channel without abandoning text entirely. `Ablation:` insert a latent-token budget into LatentWire and sweep the latent/text ratio to see whether a hybrid bridge beats all-text transport at equal bandwidth.

- **[Compress to Think, Decompress to Speak: Dual-Mode Reasoning in Transformers](https://openreview.net/forum?id=c9FF7JR8BM)**. The dual-mode setup is useful because it explicitly separates latent thinking from local text decoding, which is close to the "bridge then speak" pattern LatentWire needs. `Ablation:` add a compress phase that maps source/target context into a small latent state, then decompress only at the last few layers and compare against always-on injection.

- **[Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://openreview.net/forum?id=Oq3Xblt0x1)**. This shows that recurrence can be retrofitted into pretrained models to buy more test-time compute for reasoning without retraining from scratch. `Ablation:` use a recurrent bridge update on the transported state and test whether extra internal iterations recover failures that one-pass transport cannot.

## 2) Diffusion-style or iterative refinement for LLMs

- **[Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)**. This reframes language modeling around continuous diffusion, arguing that iterative refinement in a continuous geometry can outperform discrete diffusion baselines. `Ablation:` swap one-shot repair for a continuous denoising bridge that iteratively refines transported latent states before they enter the target decoder.

- **[Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning](https://openreview.net/forum?id=c05qIG1Z2B)**. STAR-LDM mixes latent diffusion planning with autoregressive decoding, which is a strong template for separating planning from final emission. `Ablation:` let LatentWire plan in a latent buffer for several refinement steps, then release only the final buffer to the target model.

- **[LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://openreview.net/forum?id=z5cPEZ4n6i)**. LaDiR combines a structured latent reasoning space with blockwise diffusion denoising, so the model can revise a whole chunk of thought in parallel. `Ablation:` group transported reasoning tokens into blocks and run blockwise denoising versus tokenwise correction to test whether chunked refinement is more stable.

- **[Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States](https://arxiv.org/abs/2510.11052)**. The key idea is to keep uncertain positions as belief states rather than collapsing them too early, then finalize them only after feedback. `Ablation:` preserve low-confidence transported positions as distributions over candidates and delay hard selection until after one or two repair passes.

## 3) Multimodal projector / resampler techniques

- **[PaLM2-VAdapter: Progressively Aligned Language Model Makes a Strong Vision-language Adapter](https://arxiv.org/abs/2402.10896)**. This is a strong caution that a simple perceiver-style resampler is not enough if it lacks direct supervision and staged alignment. `Ablation:` compare a direct LatentWire projector against a progressively aligned adapter trained in phases, with the same target-side budget.

- **[Honeybee: Locality-enhanced Projector for Multimodal LLM](https://arxiv.org/abs/2312.06742)**. Honeybee highlights two properties that matter for any bridge: variable token count and local-context preservation. `Ablation:` add locality-preserving pooling to the bridge and test whether keeping local neighborhoods intact helps more than global top-k compression.

- **[TokenPacker: Efficient Visual Projector for Multimodal LLM](https://arxiv.org/abs/2407.02392)**. TokenPacker uses a coarse-to-fine point-query plus region-to-point injection scheme, which is exactly the kind of structured bottleneck LatentWire can mimic. `Ablation:` replace raw token transport with a coarse query bank updated by fine regional cues, then compare against plain linear projection at the same token count.

- **[TokenFLEX: Unified VLM Training for Flexible Visual Tokens Inference](https://arxiv.org/abs/2504.03154)**. TokenFLEX is useful because it trains across variable token budgets instead of assuming a single fixed compression rate. `Ablation:` train or calibrate LatentWire across multiple bridge sizes in one run and check whether variable-budget training beats a single static budget.

## 4) Cross-model communication ideas

- **[Let Models Speak Ciphers: Multiagent Debate through Embeddings](https://arxiv.org/abs/2310.06272)**. CIPHER shows that removing token sampling and communicating in embedding space can preserve more information than natural-language debate. `Ablation:` let source and target exchange embedding-level messages before decoding, and compare that against text-mediated communication with the same token budget.

- **[KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems](https://arxiv.org/abs/2510.12872)**. KVCOMM directly addresses the multi-agent case where overlapping contexts are recomputed, and it does so by reusing and aligning KV caches online. `Ablation:` test shared KV reuse across related prompts or agents and measure whether offset alignment recovers more than independent re-encoding.

- **[Enabling Agents to Communicate Entirely in Latent Space](https://arxiv.org/abs/2511.09149)**. Interlat is a direct latent-communication baseline: last hidden states become the message, and extra compression happens entirely in latent space. `Ablation:` replace any explicit text exchange between models with latent-state passing and see whether the bridge still supports multi-step reasoning under the same bandwidth.

- **[Latent Space Communication via K-V Cache Alignment](https://arxiv.org/abs/2601.06123)**. This is the most direct recent analogue for LatentWire because it explicitly learns a shared space that aligns the KV caches of multiple models. `Ablation:` learn a shared KV bridge between source and target models and compare `aligned latent cache` vs `text-only` vs `raw cache projection`.

- **[Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](https://arxiv.org/abs/2501.02009)**. The result that concept vectors transfer across models through a simple linear map is a good warning against overcomplicating the bridge before testing linear alignment first. `Ablation:` include a linear concept-map baseline for LatentWire and check whether it already transfers useful steering signals before adding nonlinear adapters.

## 5) Highest-yield LatentWire ablations to run next

1. **Latent-thought loop vs tokenwise transport.** Compare the current bridge to a recurrent continuous-thought bridge that iteratively refines a small latent state before decoding.

2. **Soft-token bridge vs hard-token bridge.** Replace hard nearest-neighbor reconstruction with mixture-of-embeddings transport and measure whether the extra entropy helps reasoning stability.

3. **Coarse-to-fine projector vs flat projector.** Use a TokenPacker-style query bank with regional refinement and compare it to the current one-shot linear map at the same budget.

4. **Gated latent injection vs direct replacement.** Borrow the Flamingo/adapter intuition and test a near-zero gate that only lets source information in when it improves the target decoder.

5. **Latent cache sharing vs text mediation.** For multi-model or multi-agent setups, compare latent-space KV alignment and embedding messages against ordinary text handoff under identical bandwidth limits.
