# Attention Refinement / Distillation References

Goal: collect recent papers whose training signal is closer to attention refinement, iterative correction, or projector-based multimodal bridging than to direct output KL.

## Most transferable idea

The most actionable pattern for LatentWire is a **small bridge/projector trained on target-side refinement signals**: attention maps, hidden states, or a lightweight latent interface. This is a better next ablation than raw likelihood mass because it can preserve the intermediate geometry that the static bridge is missing.

## References to add next

1. **X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation** — 2025-03-08
   https://arxiv.org/abs/2503.06134
   Mechanism: a lightweight AlignNet bridge is trained with **attention distillation** into a diffusion-transformer backbone.
   LatentWire ablation: `bridge only` vs `bridge + attention distillation`.
   Cost class: **cheap/local**.

2. **MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models** — 2025-05-15
   https://arxiv.org/abs/2505.10526
   Mechanism: a lightweight projector connects the target VLM to the draft model, then **self-distilled target responses** align token predictions.
   LatentWire ablation: `transport + projector` vs `transport + projector + self-distilled teacher`.
   Cost class: **cheap/local**.

3. **Vivid-VR: Distilling Concepts from Text-to-Video Diffusion Transformer for Photorealistic Video Restoration** — 2025-08-20
   https://arxiv.org/abs/2508.14483
   Mechanism: a **control feature projector** filters artifacts and a dual-branch connector combines MLP mapping with cross-attention retrieval.
   LatentWire ablation: `single residual bridge` vs `control projector + dual-branch connector`.
   Cost class: **future-heavy**.

4. **Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers** — 2025-06-09
   https://arxiv.org/abs/2506.07986
   Mechanism: **Temperature-Adjusted Cross-modal Attention (TACA)** rebalances token-imbalance and timestep-aware attention with a parameter-efficient correction.
   LatentWire ablation: `uniform bridge strength` vs `query/timestep-weighted bridge`.
   Cost class: **cheap/local**.

5. **Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents** — 2025-09-18
   https://openreview.net/forum?id=z0WhTwZscg
   Mechanism: uses **patch-level CLIP latents** as the communicative medium and a lightweight ControlNet adaptation to bridge MLLMs and diffusion models.
   LatentWire ablation: `direct bridge` vs `shared-latent bridge`.
   Cost class: **future-heavy**.

6. **LMFusion: Adapting Pretrained Language Models for Multimodal Generation** — 2025-09-18
   https://openreview.net/forum?id=Kc1WTxZbrP
   Mechanism: freezes text modules and trains separate image modules with dedicated QKV/FFN/normalization while sharing self-attention.
   LatentWire ablation: `monolithic bridge` vs `shared-attention + modality-specific modules`.
   Cost class: **future-heavy**.

7. **Wasserstein Modality Alignment Makes Your Multimodal Transformer More Robust** — 2025-01-23
   https://openreview.net/forum?id=dbaGuiYsTl
   Mechanism: aligns modalities with a **parameter-free Wasserstein distance** instead of an extra adapter.
   LatentWire ablation: `bridge only` vs `bridge + Wasserstein alignment regularizer`.
   Cost class: **cheap/local**.

## Minimal branch to try next

1. Keep the current transport frozen.
2. Add one tiny residual projector after transport.
3. Supervise it with:
   - attention distillation,
   - target shallow hidden states,
   - or self-distilled target responses.
4. If routing helps, use a light query-conditioned router or anchor mechanism on top of the projector.

## First ablation ladder

1. transport only
2. transport + tiny projector
3. transport + projector + attention distillation
4. transport + projector + attention distillation + router

## Excluded as adjacent but less central

- **Token Distillation** and **AweDist**: strong for vocabulary-side bridge training, but they fit better in the tokenizer remapping memo than here.
- **Model-Aware Tokenizer Transfer**: useful for tokenizer adaptation, but also better kept in the output-aware/token-span alignment memo.
