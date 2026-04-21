# Tokenizer / Vocabulary Update References

Date: 2026-04-21

Scope:
- tokenizer and vocabulary adaptation
- byte / character bridges
- vocabulary expansion or replacement
- cross-tokenizer distillation
- practical tests for whether LatentWire is actually bottlenecked by tokenizer mismatch

This memo is not a claim that tokenizer mismatch is the main failure mode.
It is the next branch to test because the current LatentWire telemetry shows
that byte/span remapping is still weak and mostly diagnostic:

- `bytespan` changes very few prompts on the current GSM diagnostic slice
- the byte-stress audit only shifts a small minority of prompts
- `bytespan_module_replace` and `spanalign` are still tied on the controlled
  GSM control slice

That makes tokenizer mismatch a plausible contributor, but not yet the
dominant explanation.

## 1) Best recent sources to anchor the next branch

### Cross-tokenizer distillation and output alignment

- [Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083)
  - Cleanest direct precedent for tokenizer-agnostic distillation.
  - Use this as the main reference if the next branch adds span-level or
    byte-level supervision before the bridge.

- [Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping](https://arxiv.org/abs/2502.11104)
  - Strong for dynamic remapping when teacher and student tokenizations do not
    line up.
  - Relevant if we want a contextual remapping layer before bridge fitting.

- [Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation](https://arxiv.org/abs/2512.14954)
  - Useful for tokenizer mismatch because it focuses on likelihood scoring
    rather than assuming same-token supervision is enough.

- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)
  - Strong vocabulary-remap baseline.
  - Good for testing whether a learned token map is enough to rescue LatentWire
    before any latent transport step.

### Byte / character bridges and tokenizer-free interfaces

- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://aclanthology.org/2025.acl-long.453/)
  - Direct evidence that byte-level patching can beat rigid tokenization.
  - Good inspiration for tokenizer-independent probes and byte-oriented
    auxiliary diagnostics.

- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)
  - Direct byte-interface precedent for heterogeneous tokenizers.
  - Strong candidate if we want a byte decoder / byte probe above transported
    hidden states.

- [Length-MAX Tokenizer for Language Models](https://arxiv.org/abs/2511.20849)
  - Relevant for vocabulary redesign because it explicitly optimizes token-per-
    character efficiency.

- [KL-based self-distillation for large language models](https://arxiv.org/abs/2508.15807)
  - Useful for vocabulary expansion in frozen LLMs and for testing whether
    token knowledge can be transferred without full model surgery.

- [ByteGen: A Tokenizer-Free Generative Model for Orderbook Events in Byte Space](https://arxiv.org/abs/2508.02247)
  - Not text-domain-specific, but it is a clean byte-level reminder that
    tokenization can be removed entirely in some generation settings.

### Tokenizer / vocabulary adaptation and transfer

- [Model-Aware Tokenizer Transfer](https://openreview.net/forum?id=IyV1QEc95F)
  - Useful if the bridge should adapt to the model, not just the string.

- [Token Distillation: Attention-Aware Input Embeddings for New Tokens](https://openreview.net/forum?id=n20ml5nGEo)
  - Good source for how to initialize new vocabulary entries from existing
    tokens rather than using random embeddings.

- [FLEXITOKENS: Flexible Tokenization for Evolving Language Models](https://openreview.net/forum?id=HrrT7arjiR)
  - Relevant if we later test learnable token boundaries instead of fixed
    segmentation.

- [MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies](https://arxiv.org/abs/2502.00894)
  - Good for morphology-aware tokenizer stress tests and token-family metrics.

## 2) Runnable hooks already in the repo

These are the concrete entry points that can test the tokenizer/vocab hypothesis
without adding a new codebase.

- `scripts/analyze_byte_alignment.py`
  - audits whether span alignment and byte alignment disagree on matched
    prompts
  - outputs JSONL plus a markdown summary

- `scripts/build_bytes_accuracy_table.py`
  - rebuilds the paper-facing bytes-vs-accuracy frontier table
  - useful for seeing whether byte/span controls change the frontier at all

- `references/repos/tokenkit/scripts/eval_lockstep.py`
  - tokenizer-transfer evaluation hook
  - can run same-model or cross-tokenizer lockstep comparisons

- `references/repos/tokenkit/scripts/cross_tokenizer_distill.py`
  - training hook for cross-tokenizer distillation
  - the best local comparator for alignment vs transfer vs distillation

- `references/repos/KVzip/test.py`
  - query-agnostic KV compression benchmark
  - useful as a control for separating cache compression effects from tokenizer
    mismatch

- `references/repos/KVzip/eval.py`
  - evaluation hook for KVzip-style pruning ratios
  - useful to check whether apparent gains are really from cache geometry

- `references/repos/kvpress/evaluation/evaluate.py`
  - benchmark runner for multiple KV cache compression presses
  - useful as an external compression comparator when tokenizer controls are
    held fixed

## 3) Practical tests to answer the bottleneck question

### A. Cross-tokenizer controls

Run the same prompt set through:

- same tokenizer, same model
- same tokenizer, different model
- different tokenizer, same semantic content
- byte-level canonicalized prompt
- span-aligned prompt

Readout:

- if byte/span canonicalization materially improves over raw token alignment,
  tokenizer mismatch is real
- if it does not, the bottleneck is more likely bridge geometry, routing, or
  capacity

### B. Vocabulary adaptation tests

Test these vocab controls in order:

1. no vocabulary change
2. learned token alignment
3. new-token initialization from split-token means
4. distilled new-token embeddings
5. byte-level fallback probe

Readout:

- report answer accuracy
- report byte/character reconstruction error
- report token-family accuracy for digits, operators, units, punctuation, names,
  Unicode, and code-like spans

### C. Byte / character bridge tests

Add a tokenizer-independent diagnostic:

- train or freeze a byte-level probe on hidden states
- score transported states through byte likelihood as well as token likelihood
- compare token-probe vs byte-probe answer stability

Readout:

- if the byte probe works but token probe does not, the latent bridge may be
  fine and the tokenizer is the weak link
- if both fail, the bottleneck is upstream of the output interface

### D. Cross-tokenizer distillation tests

Compare:

- next-token KL
- contextual dynamic mapping
- approximate likelihood matching
- span-level teacher

Readout:

- report paired-flip delta
- report aligned span count
- report byte/character mismatch rate
- report token-family KL

## 4) Telemetry to log if this branch is real

If tokenizer mismatch is the issue, the next run should show a shift in at least
one of these:

- `chars_per_token` and `bytes_per_token` drift between source and target
- tokenizer overlap ratio on calibration prompts
- span alignment disagreement rate
- byte alignment disagreement rate
- byte-probe likelihood or reconstruction error
- token-family accuracy on digits, operators, units, names, emoji, and Unicode
- answer-flip rate under canonicalized prompts
- pairwise delta versus target-alone on controlled slices

If all of these stay flat, tokenizer mismatch is probably not the primary
limiter.

## 5) Suggested ablation stack

1. `tokenizer_overlap_control`
   - compare raw token prompts against byte/span-canonicalized prompts

2. `learned_vocab_remap`
   - TokAlign-style vocab mapping before transport

3. `byte_probe_bridge`
   - evaluate transported states through a byte decoder / byte scorer

4. `token_family_metrics`
   - digits/operators/units/names/Unicode/code-family breakdown

5. `cross_tokenizer_distill_stack`
   - contextual remapping + approximate likelihood matching + span teacher

6. `vocab_init_sweep`
   - random init vs mean-of-splits init vs distilled init for new tokens

## 6) Short readout

Current evidence says:

- byte / span mismatch is worth testing
- tokenizer mismatch is still unresolved
- tokenizer mismatch alone is not yet enough to explain LatentWire’s ceiling

So the next paper-safe claim should be narrow:

- tokenizer / vocab adaptation is a plausible missing component
- but the next branch must be evaluated with byte-probe and cross-tokenizer
  controls before it is counted as a positive-method improvement
