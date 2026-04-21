# CTPD / Preference Distillation / Token-Remapping References

Goal: collect the strongest recent sources for cross-tokenizer distillation, approximate likelihood matching, aligned-span preference transfer, and token-interaction supervision.

## Most transferable idea

The best next teacher for LatentWire is **dynamic aligned-span supervision before the bridge**: do not force exact next-token ID matching. Instead, remap teacher/student spans or tokens contextually, then distill likelihoods, preferences, hidden states, and token interactions on the aligned units.

## References to add next

1. **Cross-Tokenizer Distillation via Approximate Likelihood Matching** — 2025-03-25
   https://arxiv.org/abs/2503.20083
   Relevance: the cleanest direct alternative to next-token KL. It does pure distillation across different tokenizers by matching student predictions to teacher predictions without requiring a shared tokenizer.
   LatentWire ablation: `static KL` vs `approximate likelihood matching`.
   Cost class: **cheap/local**.

2. **CTPD: Cross Tokenizer Preference Distillation** — 2026-01-17
   https://arxiv.org/abs/2601.11865
   Relevance: transfers preference information between heterogeneous tokenizers using aligned-span projection, token-level importance sampling, and a teacher-anchored DPO-style objective. This is the strongest direct precedent for preference-style supervision in the cross-tokenizer setting.
   LatentWire ablation: `uniform span loss` vs `aligned-span preference loss`.
   Cost class: **cheap/local**.

3. **Model-Aware Tokenizer Transfer** — 2025-10-24
   https://arxiv.org/abs/2510.21954
   Relevance: goes beyond embedding similarity and distills inter-token communication patterns with an attention-influence objective. This is useful when raw tokenizer transfer is too local and needs model-level dynamics.
   LatentWire ablation: `token-only remap` vs `attention-influence teacher`.
   Cost class: **cheap/local**.

4. **Token Distillation: Attention-Aware Input Embeddings for New Tokens** — 2026-01-26
   https://openreview.net/forum?id=n20ml5nGEo
   Relevance: learns new token embeddings by distilling representations obtained under the original tokenization. It is a compact recipe for vocabulary-side adaptation that preserves contextual behavior.
   LatentWire ablation: `random new-token init` vs `distilled token init`.
   Cost class: **cheap/local**.

5. **DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation** — 2026-02-25
   https://arxiv.org/abs/2602.21669
   Relevance: combines entropy-based token weighting with Soft-DTW alignment over embeddings and final hidden states. This is the strongest dynamic remapping objective in the set when sequence lengths and boundaries drift.
   LatentWire ablation: `aligned-span KD` vs `Soft-DTW + weighted KD`.
   Cost class: **cheap/local**.

6. **Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions** — 2026-02-10
   https://arxiv.org/abs/2602.09483
   Relevance: argues that static next-token alignment misses the important interaction structure. It aligns vision-instruction interactions and intra-response token transitions instead of only marginals.
   LatentWire ablation: `next-token-only teacher` vs `token-interaction teacher`.
   Cost class: **cheap/local**.

7. **Vision Language Model Distillation Using Partial Information Decomposition** — 2025-06-11
   https://openreview.net/forum?id=caBO989n7l
   Relevance: decomposes mutual information into unique, redundant, and synergistic components, explicitly targeting multimodal interaction structure rather than output KL alone.
   LatentWire ablation: `logit/hidden KD` vs `synergy-aware KD`.
   Cost class: **future-heavy**.

## Minimal branch to try next

Implement a **dynamic aligned-span teacher** before the bridge:

1. Convert teacher and student text to character-offset spans.
2. Align spans with:
   - Hungarian matching if you want a simple one-to-one remap, or
   - monotone DP / Soft-DTW if order is mostly preserved but boundaries drift.
3. Distill on the aligned spans:
   - span-pooled hidden states,
   - span-level likelihoods/logits,
   - optional token/span interaction or affinity matrices.

Minimal objective:

\[
\mathcal{L}
= \lambda_{KL}\,\mathrm{KL}(p_T \| p_S)
+ \lambda_h \lVert h_T - h_S \rVert_2^2
+ \lambda_a \lVert A_T - A_S \rVert_1
\]

## Why this is materially different from static top-k KL

- Static top-k KL still assumes the same output positions should match.
- Dynamic span remapping changes the correspondence itself, which is what fails when tokenizers split the same text differently.
- Preference and interaction losses let you supervise the teacher/student relation through aligned spans, not exact token IDs.

## Recommended first ablation ladder

1. Static next-token KL
2. Aligned-span KL only
3. Aligned-span KL + hidden-state loss
4. Aligned-span KL + hidden-state + interaction loss
5. If available, replace span KL with preference-style aligned-span loss

## Excluded as adjacent but less direct

- **AweDist**: very relevant for new-token initialization, but redundant once Token Distillation is included.
- **TE-VLM**: useful for information-flow regularization, but it is less directly tied to cross-tokenizer span remapping than the selected references.
