# Tokenizer / Vocabulary Interface Memo

## Bottom line
The most useful next move for LatentWire is a **byte-level or shared-span interface ablation** before any more embedding-only alignment work. The recent 2025-2026 literature is converging on the same failure mode: tokenizer mismatch is not just an embedding problem, it is a **sequence alignment + vocabulary mismatch + credit assignment** problem. A shared byte/span interface is the cleanest way to test whether LatentWire can still transfer signal when token IDs do not match.

## Primary sources
- [Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083)
  Pure cross-tokenizer distillation; explicitly handles large tokenizer/vocabulary mismatch and shows transfer to byte-level tokenization.
- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)
  Learns a token-ID alignment / vocabulary replacement path, then fine-tunes after remapping.
- [FLEXITOKENS: Flexible Tokenization for Evolving Language Models](https://arxiv.org/abs/2507.12720)
  Learnable byte-level tokenizer with variable-length segmentation; useful as a tokenizer-adaptation baseline.
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)
  Latest byte-level transfer baseline; directly supports the idea that bytes are the most stable shared interface.
- [CTPD: Cross Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865)
  Preference transfer across heterogeneous tokenizers via aligned span projection.
- [Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation](https://arxiv.org/abs/2512.14954)
  Likelihood computation under mismatched BPEs; useful for evaluating whether LatentWire can compare teacher/student distributions without exact token overlap.
- [FOCUS: Effective Embedding Initialization for a New Tokenizer](https://arxiv.org/abs/2305.14481)
  Older but still relevant vocabulary initialization baseline when partial overlap exists.

## Concrete ablations to run
1. **Byte fallback bridge**
   Replace only the interface layer with byte-level targets/decoders and keep the core bridge unchanged. This isolates whether the failure is at tokenization or at representation transport.
2. **Shared-span supervision**
   Recompute distillation targets on character/byte spans instead of token IDs. Compare exact token-level KD vs span-aligned KD vs byte-level KD.
3. **Vocabulary remap init**
   Start from TokAlign-style one-to-one vocab remapping, then freeze the backbone and train only a lightweight bridge. This tests whether alignment helps before any transport learning.
4. **Byte-level decoder head**
   Attach a small byte decoder head to the student and distill through the byte interface, mirroring the 2026 BLD line of work.
5. **Span projection vs token projection**
   Replace token-to-token matching with aligned-span projection for both logits and hidden states. Measure whether the gain is from better supervision geometry rather than more capacity.
6. **Tokenizer transfer with frozen bridge**
   Transfer the tokenizer only, keep the model weights frozen, and evaluate if LatentWire still preserves usable signal. This is a cheap way to separate tokenizer effects from model effects.
7. **Compression sensitivity sweep**
   Sweep vocabulary size / byte compression ratio / average tokens per character and log accuracy, bytes generated, and KV footprint. This makes the paper interpretable later.
8. **Mixed-interface distillation**
   Combine same-tokenizer and cross-tokenizer supervision in one run, but separate the loss terms and log them independently. This mirrors the best-performing direction in the 2025-2026 cross-tokenizer papers.

## Recommendation for first implementation
Implement **shared-span / byte-level interface distillation first**.

Reason: it is the lowest-assumption test, it directly addresses both vocabulary mismatch and sequence misalignment, and it gives a clean yes/no answer for whether LatentWire needs a better interface or a better transport mechanism. If this fails, the next step should be token remapping; if it works, the bridge design can be refined on top of a stable interface.

## What to log for interpretability
- Average tokens per character and bytes generated per sample.
- Alignment error between teacher/student spans.
- Distillation loss broken into same-token vs cross-token components.
- KV/cache footprint per interface choice.
- Per-layer transport quality, not just final task accuracy.
