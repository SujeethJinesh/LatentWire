# Multimodal Bridge / Projector Alignment

Goal: identify the smallest bridge that is still materially richer than the current tiny residual adapters.

## Most transferable idea

Use a **tiny projector supervised by target-side refined embeddings or logits**, optionally with a lightweight router. This is the closest analog to a bridge that can still fit the current LatentWire flow.

## References to add next

Web check on 2026-04-20: these entries were kept because I found primary
arXiv/OpenReview-style sources for the exact titles or close title variants.

1. **X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation** — 2025-03-08  
   https://arxiv.org/abs/2503.06134  
   Why it matters: a lightweight AlignNet bridge plus attention distillation transfers multimodal understanding into a diffusion transformer.  
   Ablation it suggests: `output loss only` vs `attention-distilled bridge`.

2. **UniCrossAdapter: Multimodal Adaptation of CLIP for Radiology Report Generation** — 2025-03-20  
   https://arxiv.org/abs/2503.15940  
   Why it matters: distributes lightweight adapters across modalities and their interaction while freezing the base.  
   Ablation it suggests: `monolithic bridge` vs `distributed adapters`.

3. **MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models** — 2025-05-15  
   https://arxiv.org/abs/2505.10526  
   Why it matters: lightweight projector plus self-distilled target responses is a strong template for a tiny bridge with better supervision.  
   Ablation it suggests: `bridge only` vs `bridge + self-distilled teacher`.

4. **MedBridge: Bridging Foundation Vision-Language Models to Medical Image Diagnosis in Chest X-Ray** — 2025-05-27  
   https://arxiv.org/abs/2505.21698  
   Why it matters: adds a small query encoder and MoE on top of frozen VLMs.  
   Ablation it suggests: `single bridge` vs `query-encoded MoE bridge`.

5. **BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models** — 2025-08-09  
   https://arxiv.org/abs/2508.06895  
   Why it matters: directly supervises the projector with refined internal embeddings and logit-space matching.  
   Ablation it suggests: `logit-only supervision` vs `hidden-state + logit supervision`.

6. **Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents** — 2025-08-08  
   https://arxiv.org/abs/2508.05954  
   Why it matters: uses patch-level CLIP latents as a shared latent interface, which is a clean bridge analogue.  
   Ablation it suggests: `direct bridge` vs `shared-latent bridge`.

7. **AttAnchor: Guiding Cross-Modal Token Alignment in VLMs with Attention Anchors** — 2025-09-27  
   https://arxiv.org/abs/2509.23109  
   Why it matters: attention anchors are a cheap routing primitive for grouping semantically related tokens before alignment.  
   Ablation it suggests: `no routing` vs `anchor-based routing`.

## Minimal recipe

1. Keep transport frozen.
2. Add one tiny residual projector after transport.
3. Supervise it with:
   - target shallow hidden states,
   - target logits,
   - optionally attention distillation or a query router.

## Recommended ablation ladder

1. bridge only
2. bridge + target hidden-state loss
3. bridge + hidden-state + logit loss
4. bridge + hidden-state + logit + router
