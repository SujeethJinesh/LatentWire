# Cross-Tokenizer / Byte-Level Alignment

Goal: identify the next branch for LatentWire when the bottleneck is upstream token/span correspondence rather than bridge capacity.

## Most transferable idea

Use **dynamic span remapping before the bridge**, then supervise on aligned spans with likelihood and hidden-state losses. This is cheaper than a full byte-level interface and more likely to survive tokenizer mismatch.

## References to add next

Web check on 2026-04-20: these entries were kept because I found primary
arXiv/OpenReview sources for the exact titles or close title variants.

1. **Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping** — 2025-02-16  
   https://arxiv.org/abs/2502.11104  
   Why it matters: contextual dynamic mapping fixes sequence misalignment and vocabulary mismatch better than static positionwise KL.  
   Ablation it suggests: `static next-token KL` vs `contextual remapping`.

2. **Cross-Tokenizer Distillation via Approximate Likelihood Matching** — 2025-03-25  
   https://arxiv.org/abs/2503.20083  
   Why it matters: pure distillation without a next-token objective is a cleaner teacher when tokenizers differ.  
   Ablation it suggests: `next-token KL` vs `approximate likelihood matching`.

3. **TokAlign: Efficient Vocabulary Adaptation via Token Alignment** — 2025-06-04  
   https://arxiv.org/abs/2506.03523  
   Why it matters: learns a one-to-one vocabulary remap from token co-occurrences, then transfers token-level knowledge after unifying vocabularies.  
   Ablation it suggests: `no vocab remap` vs `learned vocab remap`.

4. **FLEXITOKENS: Flexible Tokenization for Evolving Language Models** — 2025  
   https://openreview.net/forum?id=HrrT7arjiR  
   Why it matters: byte-level LMs with learnable tokenizers avoid rigid fixed segmentation.  
   Ablation it suggests: `fixed tokenizer` vs `learnable token boundaries`.

5. **Model-Aware Tokenizer Transfer** — 2025-09-20  
   https://openreview.net/forum?id=IyV1QEc95F  
   Why it matters: uses inter-token communication patterns in attention layers to adapt pretrained models to new tokenizers.  
   Ablation it suggests: `token-only transfer` vs `attention-influence transfer`.

6. **Token Distillation: Attention-Aware Input Embeddings for New Tokens** — 2026-01-26  
   https://openreview.net/forum?id=n20ml5nGEo  
   Why it matters: distills representations from the original tokenization to initialize embeddings for new tokens.  
   Ablation it suggests: `random/new-token init` vs `distilled new-token init`.

## Minimal recipe

1. Convert teacher and student text to character-offset spans.
2. Align spans with a cheap solver:
   - Hungarian if you want one-to-one remapping.
   - Monotone DP or Soft-DTW if order is mostly preserved.
3. Distill on aligned spans:
   - span-pooled hidden states,
   - span-level logits,
   - optional token/span affinity matrices.

Minimal objective:

\[
\mathcal{L}
= \lambda_{KL}\,\mathrm{KL}(p_T \| p_S)
+ \lambda_h \lVert h_T - h_S \rVert_2^2
+ \lambda_a \lVert A_T - A_S \rVert_1
\]

## Failure mode this should fix

Raw prompt overlap fails when the same text is split into different numbers and boundaries of subtokens, so same-position supervision is not the same semantic unit.
