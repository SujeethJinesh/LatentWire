# Unsaturated Branch Revival Refs (2026-04-21)

Purpose: mark which old ideas are saturated versus worth reviving so we stop
going in circles while still keeping the strongest compositions alive.

## Saturated Or Near-Saturated

- static grouped transport variants on the exact same-pair contract
- confidence-only routing and current frontier / stop heuristics
- stronger same-family teacher variants on frozen GSM8K32
- byte-only alignment as a same-pair rescue
- first SAE-style sparse bridge on the frozen contract
- naive fixed gauge wrappers on the live dynalign residual lane

## Still Open

- quotient-aware canonicalization before or around the live lane
- GPA/shared canonical hubs combined with sparse shared dictionaries
- byte/sequence-aligned sidecars on top of the shared basis
- residual repair on the live dynalign lane after preserving dominant
  directions
- cross-tokenizer / byte-level interface control on genuinely mismatched pairs

## Revival / Composition Ideas

1. `quotient-aware canonicalization -> GPA hub -> residual repair`
   Anchors:
   - RECON — https://openreview.net/forum?id=bpWzTPDybh
   - Multi-Way Representation Alignment — https://arxiv.org/abs/2602.06205
   - Preserve-Then-Quantize — https://arxiv.org/abs/2602.02001
   - ResQ — https://arxiv.org/abs/2412.14363

2. `shared sparse basis -> byte/sequence sidecar`
   Anchors:
   - Universal Sparse Autoencoders — https://arxiv.org/abs/2502.03714
   - SPARC — https://arxiv.org/abs/2507.06265
   - LUCID-SAE — https://arxiv.org/abs/2602.07311
   - Cross-Tokenizer LLM Distillation through a Byte-Level Interface — https://arxiv.org/abs/2604.07466
   - Delta-Crosscoder — https://arxiv.org/abs/2603.04426

3. `stable core bridge -> budgeted dynamic gate / residual bank`
   Anchors:
   - C2C — https://arxiv.org/abs/2510.03215
   - KVComm — https://arxiv.org/abs/2510.03346
   - GaugeKV — https://openreview.net/forum?id=rSxYPLzyBu
   - R2Q — https://arxiv.org/abs/2511.21736

## Recommended Order

1. Adaptive canonicalization on top of `dynalign_module_replace_residrank16`
2. If that fails, eigenspace-aware residual repair on the live lane
3. Then revive the shared-basis + sidecar composition on a mismatched pair

## Current Read

- Do not spend more time on static teacher variants, confidence-only routing,
  or fixed gauge wrappers.
- The highest-signal revival is adaptive canonicalization plus residual repair,
  with shared-basis + sidecar as the broader low-shot fallback story.
