# Anchor-Preserving Codebook Tail Refs (2026-04-21)

Purpose: capture the strongest codebook-style tail-compression references now
that the toy preserve-top-k branch looks promising but the first naive codebook
tail is still not additive.

## Strongest Sources

1. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: additive codebooks at extreme compression are the clearest
   reference for multi-stage residual coding.

2. CommVQ
   Link: https://arxiv.org/abs/2506.18879
   Why it matters: commutative vector quantization for KV-cache style states is
   directly relevant to cross-model message transport.

3. QINCo
   Link: https://arxiv.org/abs/2401.14732
   Why it matters: later residual codebooks should depend on earlier residuals,
   not be fit independently.

4. QINCo2
   Link: https://arxiv.org/abs/2501.03078
   Why it matters: improved implicit residual codebooks and better stagewise
   compression/search behavior.

5. ESC-MVQ
   Link: https://arxiv.org/abs/2504.11709
   Why it matters: multi-codebook semantic communication rather than one-shot
   discretization.

6. MOC-RVQ
   Link: https://arxiv.org/abs/2401.01272
   Why it matters: multilevel residual vector quantization is the nearest
   communications-side analogue of a preserve-anchor codec tail.

7. RQT
   Link: https://aclanthology.org/2025.findings-acl.554/
   Why it matters: hierarchical residual quantization for multi-model
   compression supports stagewise rather than single-shot tail coding.

## Exact Next Ablations

1. `saliency-preserve codebook tail`
   Why now: keep the preserved anchor idea, but move the tail from uniform or
   naive codebook compression to a saliency-aware residualized codebook.

2. `rotation-before-codebook`
   Why now: compare identity, orthogonal, Hadamard, and DCT before coding the
   tail so the codebook does not inherit a bad basis for free.

3. `stagewise residual codebooks`
   Why now: only if one-stage codebook tails still underperform the preserved
   anchor baseline.

## Interpretable Telemetry

- selected anchor atoms
- codebook perplexity and dead-code rate
- preserve MSE, tail MSE, and full reconstruction MSE
- bytes proxy and index-bit budget
- help / harm vs the preserved-anchor baseline

## Current Read

- The toy preserve-top-k result says anchors matter, but the first naive
  codebook tail still underperforms the simpler preserved-anchor baseline.
- If the real saliency-preserve residual row fails, the cleanest next codec
  branch is anchor-preserving codebook tails rather than another blind dense
  residual tweak.
