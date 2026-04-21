# KV-Cache Communication / Compression

Goal: capture the most relevant recent work on KV retention, compression, canonicalization, and retrieval-style cache reuse.

## Most transferable idea

Use **head-aware canonicalization or retrieval-aware retention** as the KV-side analog of a bridge: preserve the heads and tokens that matter, compress the rest, and test whether the resulting signal is better than static cache heuristics.

## References to add next

Web check on 2026-04-20: entries below were kept because they have primary
arXiv/OpenReview sources. A previously proposed `GaugeKV` pointer was removed
from the citeable list because I did not find a primary source for that exact
title during this pass.

1. **ContextKeeper: Head-Specific KV Cache Retention for Long-Context LLM Inference** — OpenReview ICLR 2026 submission  
   https://openreview.net/forum?id=7zQy4iHoZ8  
   Why it matters: head-specific retention distinguishes context-anchored heads from locality heads.  
   Ablation it suggests: `global retention` vs `head-specific retention`.

2. **RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation** — OpenReview ICLR 2026 submission  
   https://openreview.net/forum?id=F7kDkYjBVa  
   Why it matters: combines compression with retrieval so evicted KV can be recovered when needed.  
   Ablation it suggests: `compression only` vs `compression + retrieval`.

3. **KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction** — 2025-10-06  
   https://arxiv.org/abs/2510.05373  
   Why it matters: rotation plus lightweight linear correction explicitly repairs attention errors after compression.  
   Ablation it suggests: `quantization only` vs `quantization + correction`.

4. **KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity** — 2025-12-05  
   https://arxiv.org/abs/2512.05916  
   Why it matters: directly optimizes the low-rank approximation of the attention matrix rather than the cache tensors in isolation.  
   Ablation it suggests: `KV-only compression` vs `attention-fidelity compression`.

## Internal ablation idea without a cite yet

Gauge-style canonicalization is still a sensible LatentWire ablation because
our layer/head spaces appear to have repeated orientation and permutation
symmetries, but it should be cited through a verified source before entering
paper prose.

## Minimal recipe

1. Start from the current cache path.
2. Test one head-aware retention or canonicalization rule.
3. Compare:
   - full KV,
   - compressed KV,
   - compressed KV plus small correction / retrieval.

## Failure mode this should fix

Generic cache heuristics drop middle-context tokens or smear head-specific structure, which hurts multi-turn fidelity even when one-turn accuracy looks fine.
