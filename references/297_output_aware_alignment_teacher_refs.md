# Output-Aware Dynamic Alignment Teachers

Goal: identify the next teacher-side branch for LatentWire once static top-k KL and local dynalign have saturated.

## Most transferable idea

Use **dynamic token/span remapping before the bridge**, then distill on aligned spans with likelihood, hidden-state, and optional interaction losses. This is materially different from static next-token KL because the correspondence itself changes with context.

## Best recent primary sources

1. **Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping** — 2025-02-16  
   https://arxiv.org/abs/2502.11104  
   Relevance: the cleanest direct precedent for contextual remapping when teacher/student tokenizers disagree. It explicitly targets sequence misalignment and vocabulary mismatch.  
   LatentWire ablation: `static KL` vs `contextual remapping`.

2. **Cross-Tokenizer Distillation via Approximate Likelihood Matching** — 2025-03-25  
   https://arxiv.org/abs/2503.20083  
   Relevance: removes the need for a shared tokenizer by matching teacher/student likelihoods directly. This is a stronger teacher than prompt-local next-token KL.  
   LatentWire ablation: `next-token KL` vs `approximate likelihood matching`.

3. **Model-Aware Tokenizer Transfer** — 2025-10-24  
   https://arxiv.org/abs/2510.21954  
   Relevance: distills inter-token communication patterns via attention influence modeling, which is closer to a dynamic teacher than a static output loss.  
   LatentWire ablation: `token-only remap` vs `attention-influence teacher`.

4. **TokAlign: Efficient Vocabulary Adaptation via Token Alignment** — 2025-06-04  
   https://arxiv.org/abs/2506.03523  
   Relevance: learns a one-to-one vocabulary remap from token co-occurrences, then transfers token-level knowledge after unifying vocabularies. This is a good pre-bridge alignment baseline.  
   LatentWire ablation: `no vocab remap` vs `learned vocab remap`.

5. **CTPD: Cross Tokenizer Preference Distillation** — 2026-01-17  
   https://arxiv.org/abs/2601.11865  
   Relevance: aligns teacher/student tokens to shared character-level spans and adds token-level importance sampling for preference transfer. Useful when the teacher signal is preference-like rather than pure likelihood.  
   LatentWire ablation: `uniform span loss` vs `importance-weighted span loss`.

6. **Token Distillation: Attention-Aware Input Embeddings for New Tokens** — 2026-01-26  
   https://openreview.net/forum?id=n20ml5nGEo  
   Relevance: distills representations obtained under the original tokenization to initialize new tokens. This is a compact target-side refinement recipe for a tiny bridge.  
   LatentWire ablation: `random token init` vs `distilled token init`.

7. **DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation** — 2026-02-25  
   https://arxiv.org/abs/2602.21669  
   Relevance: combines entropy-based token weighting with Soft-DTW alignment over embedding and final hidden-state layers. This is the strongest dynamic remapping recipe in the set.  
   LatentWire ablation: `aligned-span loss` vs `Soft-DTW remapping + weighted KD`.

## Minimal branch to try next

Implement a **span-level teacher** before the bridge:

1. Build character-offset spans for teacher and student tokenizations.
2. Align spans with either:
   - Hungarian matching if one-to-one remapping is acceptable, or
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

## What failure mode this is meant to fix

Raw prompt overlap fails when the same text is split into different numbers and boundaries of subtokens, so same-position supervision is not the same semantic unit.

## Recommended first ablation ladder

1. Static next-token KL
2. Aligned-span KL only
3. Aligned-span KL + hidden-state loss
4. Aligned-span KL + hidden-state + interaction loss

## Excluded as adjacent but less direct

- **MASSV** and **BASIC**: useful for bridge supervision, but they are more about projector/bridge training than token/span remapping.
- **AweDist**: closely related to Token Distillation, but redundant once Token Distillation is included.
