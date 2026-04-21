# Dynamic Alignment Bridge Memo

Goal: identify the next most implementable direction for LatentWire when static bridge capacity has saturated.

## Best current read

The strongest next step is **dynamic token/span remapping before the bridge**, not a larger static bridge. The recurring pattern in recent work is:

- static top-k / position KL fails under tokenizer mismatch,
- span-level or contextual remapping recovers correspondence,
- output-side supervision should match on spans, hidden states, and interactions, not only next-token probabilities.

## References to add next

1. **Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping** — 2025-02-16  
   https://arxiv.org/abs/2502.11104  
   Why it matters: contextual remapping fixes sequence misalignment and vocab mismatch better than static positionwise KL.  
   Ablation it suggests: `static KL` vs `contextual dynamic mapping`.

2. **Cross-Tokenizer Distillation via Approximate Likelihood Matching** — 2025-03-25  
   https://arxiv.org/abs/2503.20083  
   Why it matters: likelihood matching without a shared tokenizer is a cleaner teacher than prompt-local output KL.  
   Ablation it suggests: `next-token KL` vs `approximate likelihood matching`.

3. **TokAlign: Efficient Vocabulary Adaptation via Token Alignment** — 2025-06-04  
   https://arxiv.org/abs/2506.03523  
   Why it matters: learns a one-to-one vocab remap from token co-occurrences, then transfers token-level knowledge after unifying vocabularies.  
   Ablation it suggests: `no vocab remap` vs `learned vocab remap`.

4. **MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models** — 2025-05-15  
   https://arxiv.org/abs/2505.10526  
   Why it matters: lightweight projector + self-distilled target responses is a strong template for a tiny bridge with teacher supervision.  
   Ablation it suggests: `bridge only` vs `bridge + self-distilled teacher`.

5. **BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models** — 2025-08-09  
   https://arxiv.org/abs/2508.06895  
   Why it matters: directly supervises the projector with refined internal embeddings and logit-space matching.  
   Ablation it suggests: `logit-only teacher` vs `hidden-state + logit teacher`.

6. **AttAnchor: Guiding Cross-Modal Token Alignment in VLMs with Attention Anchors** — 2025-09-27  
   https://arxiv.org/abs/2509.23109  
   Why it matters: attention anchors are a cheap routing primitive for grouping semantically related tokens before alignment.  
   Ablation it suggests: `no routing` vs `query-conditioned anchor routing`.

7. **X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation** — 2025-03-08  
   https://arxiv.org/abs/2503.06134  
   Why it matters: attention distillation into a lightweight bridge is a direct analogue for a bridge that learns from target attention geometry.  
   Ablation it suggests: `output loss only` vs `attention distillation bridge`.

8. **CTPD: Cross Tokenizer Preference Distillation** — 2026-01-17  
   https://arxiv.org/abs/2601.11865  
   Why it matters: aligned span projection plus token-level importance sampling makes cross-tokenizer supervision practical.  
   Ablation it suggests: `uniform span loss` vs `importance-weighted aligned-span loss`.

## Most transferable next branch

Implement **dynamic span remapping before the bridge**, then supervise a tiny bridge/projector with:

- aligned-span KL,
- span-pooled hidden-state loss,
- optional token-interaction / affinity loss.

Cheapest solver choices:

- **Hungarian** if we want a simple one-to-one span map.
- **Monotone DP / Soft-DTW** if order is mostly preserved but boundaries drift.

Minimal objective:

\\[
\\mathcal{L}
= \\lambda_{KL}\\,\\mathrm{KL}(p_T \\| p_S)
+ \\lambda_h \\lVert h_T - h_S \\rVert_2^2
+ \\lambda_a \\lVert A_T - A_S \\rVert_1
\\]

where the teacher/student correspondence is computed dynamically, not by raw position.

## One failure mode this should fix

Raw prompt overlap fails when the same text is split into different numbers and boundaries of subtokens, so “same position” is not the same semantic unit.

## Recommended first ablation ladder

1. Static next-token KL
2. Aligned-span KL only
3. Aligned-span KL + hidden-state loss
4. Aligned-span KL + hidden-state + interaction loss

