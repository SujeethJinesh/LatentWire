# Tokenizer / Vocabulary Blockers for LatentWire

Date: 2026-04-21

Scope:
- tokenizer mismatch and vocabulary transfer
- byte-level or span-level shared interfaces
- embedding initialization for new tokenizers
- byte/patch latent LMs
- cross-tokenizer model reuse for communication

## Bottom line

The strongest signal from the recent literature is that tokenizer mismatch is not just a lexical nuisance. It is a supervision-alignment problem and, in many settings, a vocabulary-geometry problem. For LatentWire, the cleanest first question is whether cross-model communication improves more from a shared byte/span interface than from a richer bridge.

Inference from the sources: if the current LatentWire runs are still brittle under tokenization changes, the highest-yield next step is to canonicalize communication at bytes or aligned spans before investing more in bridge capacity.

## Primary sources

### Byte-level and patch-based LMs

- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://aclanthology.org/2025.acl-long.453/) and the official code repo [bowang-lab/dnaBLT](https://github.com/bowang-lab/dnaBLT)
  - Strong evidence that a byte-native model can use dynamic patches as the main computation unit rather than token IDs.
  - Use this as the closest patch/byte latent-LM baseline for a tokenizer-agnostic interface.

- [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626)
  - Canonical token-free baseline showing that standard Transformers can work directly on bytes with minimal architectural change.
  - Useful as the simplest shared-substrate control when token overlap disappears.

- [EvaByte: Efficient Byte-level Language Models at Scale](https://github.com/OpenEvaByte/evabyte)
  - Recent byte-level LM with multibyte prediction and efficient attention.
  - Good systems reference for decoding cost, byte throughput, and byte-native generation quality.

- [Length-MAX Tokenizer for Language Models](https://arxiv.org/abs/2511.20849)
  - Shows that optimizing token length directly can reduce sequence length and KV footprint without sacrificing downstream quality.
  - Useful when the issue is not full token freedom but better segmentation efficiency.

### Cross-tokenizer distillation and scoring

- [Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083)
  - Cleanest cross-tokenizer distillation baseline for transferring knowledge when teacher and student tokenizers differ.
  - Important because it treats tokenizer mismatch as the central obstacle rather than a nuisance term.

- [Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping](https://arxiv.org/abs/2502.11104)
  - Adds contextual sequence alignment plus dynamic vocabulary mapping.
  - Useful when static one-to-one token remapping is too weak.

- [Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation](https://arxiv.org/abs/2512.14954)
  - Gives a principled way to score likelihood across mismatched BPE vocabularies.
  - Good for evaluator-side alignment when exact token overlap is unavailable.

- [CTPD: Cross Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865)
  - Extends cross-tokenizer transfer to preference alignment with aligned span projection and token-level credit assignment.
  - Relevant if LatentWire needs a preference-style or reranking-style communication target.

- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)
  - Direct byte-level interface for heterogeneous teacher/student tokenizers.
  - Strongest recent “shared substrate” baseline for LatentWire-style cross-model communication.

### Tokenizer and vocabulary transfer

- [Zero-Shot Tokenizer Transfer](https://arxiv.org/abs/2405.07883) and the official code repo [bminixhofer/zett](https://github.com/bminixhofer/zett)
  - Establishes tokenizer swap as a first-class problem and uses a hypernetwork to predict embeddings for a new tokenizer.
  - Strong baseline when the goal is to preserve the backbone while changing the surface vocabulary.

- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)
  - Learns a token-ID alignment / vocabulary replacement path and then progressively fine-tunes.
  - Best direct baseline for one-to-one vocab remapping before any latent transport logic.

- [Model-Aware Tokenizer Transfer](https://arxiv.org/abs/2510.21954)
  - Distills attention influence patterns, not just embedding similarity, into a new-tokenizer warm start.
  - Stronger than lexical heuristics when the model’s internal token interactions matter.

- [FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models](https://arxiv.org/abs/2305.14481)
  - Classic embedding-initialization reference for a new tokenizer.
  - Useful as the “cheap initialization” baseline before more elaborate tokenizer transfer.

- [Token Distillation: Attention-Aware Input Embeddings for New Tokens](https://openreview.net/forum?id=n20ml5nGEo)
  - Learns new-token embeddings by distilling behavior from the original tokenization.
  - Good baseline when the surface vocabulary changes but the backbone stays fixed.

## Concrete LatentWire ablations

1. **Byte/span canonical bridge**
   Replace token-ID supervision with byte-level or shared character-span supervision, keeping the bridge architecture fixed. This is the cleanest test of whether the bottleneck is tokenization or transport.

2. **Vocab remap before transport**
   Run a TokAlign-style vocabulary replacement or ZeTT-style tokenizer transfer first, then fit the same LatentWire bridge on top. If this closes most of the gap, the bridge was not the main problem.

3. **Byte decoder head**
   Attach a lightweight byte-level decoder head to the transported latent state and compare token-probe vs byte-probe accuracy. This separates output-interface failure from hidden-state transport failure.

4. **Embedding-init ladder**
   Compare random init, FOCUS-style init, Token Distillation, and Model-Aware Tokenizer Transfer for any new tokens. Log convergence speed and whether the new tokenizer preserves downstream behavior after a fixed training budget.

5. **Cross-tokenizer scoring controls**
   Evaluate the same source-target pair under next-token KL, ALM, CDM, and BPE-aware likelihood scoring. If the ranking changes materially, the current evaluator is tokenizer-sensitive.

6. **Token-family stress test**
   Break results out by digits, operators, units, punctuation, names, Unicode, and code-like spans. Tokenizer blockers usually show up first in these families, not in aggregate accuracy.

7. **Byte-native vs token-native relay**
   Compare a byte-native relay path against a standard token-native relay at matched byte budget and matched wall-clock budget. This isolates whether tokenization is the limiter or just an efficient compression prior.

## What to log

- accuracy and exact-match under matched byte budget
- average bytes per communication step
- token fragmentation rate and chars-per-token drift
- cross-tokenizer KL or likelihood mismatch
- byte-probe vs token-probe reconstruction quality
- token-family accuracy on numerals, punctuation, code, and Unicode
- latency and KV footprint for each interface choice

## Recommendation

Start with the byte/span canonical bridge and vocab remap before transport. Those two ablations are the cheapest way to answer whether LatentWire needs a better interface or just a better bridge.
