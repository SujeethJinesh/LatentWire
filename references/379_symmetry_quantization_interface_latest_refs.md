# Symmetry / Quantization / Interface References for LatentWire

Web check: 2026-04-21. Primary-source memo only, biased toward 2025-2026. I included recent workshop/submission links when they were the only primary sources directly targeting Transformer gauge structure or gauge-aware KV compression; use those as theory leads, not settled baselines.

## Sources

### Gauge, symmetry, canonicalization

| Source | Status | LatentWire read |
|---|---|---|
| [Complete Characterization of Gauge Symmetries in Transformer Architectures](https://openreview.net/forum?id=KrkbYbK0cH) | NeurReps 2025 proceedings | Establishes the head-wise Transformer gauge group `((GL(d_k))^h x (GL(d_v))^h) ⋊ S_h` and argues LayerNorm preserves it. Raw Q/K/V coordinates are therefore non-identifiable up to continuous basis changes and head permutation; a bridge should be gauge-fixed or quotient-aware before claiming alignment. |
| [Gauge Fiber Bundle Geometry of Transformers](https://openreview.net/forum?id=YC9O7OyLFK) | ICLR 2026 withdrawn submission | Treats Transformer parameter space as a principal bundle with gauge orbits as fibers and function-changing directions as horizontals. This is the cleanest recent argument for splitting bridge updates or diagnostics into gauge drift vs task-relevant drift. |
| [Adaptive Canonicalization with Application to Invariant Anisotropic Geometric Networks](https://arxiv.org/abs/2509.24886) | arXiv 2025 | Shows fixed canonicalization can be discontinuous, and proposes adaptive canonicalization with continuity and approximation guarantees. For LatentWire: a fixed SVD/sort rule may be too brittle; an input-conditional canonicalizer is plausible. |
| [Rethinking Diffusion Models with Symmetries through Canonicalization with Applications to Molecular Graph Generation](https://arxiv.org/abs/2602.15022) | arXiv 2026 | Gives a quotient-space view: canonicalize first, then learn on one orbit representative; also argues optimal transport complements canonicalization. The useful transfer is methodological: collapse symmetry redundancy before solving transport or compression. |
| [Bispectral Invariants for Transformers: An Operator-Algebraic Approach](https://openreview.net/forum?id=QxVvKboznV) | NeurReps 2025 workshop submission | After canonical gauge-fixing, proposes complete invariants for residual head permutation symmetry. This is a strong cue to add gauge-free diagnostics richer than CKA or cosine. |

### Quantization geometry and protected allocation

| Source | Status | LatentWire read |
|---|---|---|
| [GaugeKV: Composable Exact KV Cache Compression](https://openreview.net/forum?id=rSxYPLzyBu) | ICLR 2026 desk-rejected submission | Reparameterizes heads into a canonical basis that is friendlier to exact and lossy KV compression, then composes with GQA/MQA/FP8. The key idea is not the benchmark number but the mechanism: canonicalize before compressing. |
| [KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](https://arxiv.org/abs/2506.08018) | arXiv 2025, AAAI 2026 oral | Allocates KV precision by gradient-based importance rather than uniform budgets. This is directly analogous to protected-channel or protected-route allocation for LatentWire messages. |
| [CommVQ: Commutative Vector Quantization for KV Cache Compression](https://arxiv.org/abs/2506.18879) | arXiv 2025, ICML 2025 poster | Learns additive VQ codebooks that commute with RoPE. The transferable idea is symmetry-compatible codebooks: interface compression should respect the target model's existing algebraic structure instead of fighting it. |
| [NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache](https://openreview.net/forum?id=boNYskaXnO) | NeurIPS 2025 poster | Uses normalize-shift-normalize plus Hadamard to align token distributions with a standard normal before VQ. This suggests that LatentWire may need canonical distribution-shaping before any packetizer or codebook. |
| [IMPQ: Interaction-Aware Layerwise Mixed Precision Quantization for LLMs](https://arxiv.org/abs/2509.15455) | arXiv 2025 | Frames mixed-precision allocation as a cooperative game and solves a binary quadratic program using Shapley-style interaction estimates. This is a strong template for route/layer budget allocation when channels interact non-additively. |

### Interface, transport, and tokenizer mismatch

| Source | Status | LatentWire read |
|---|---|---|
| [Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch](https://arxiv.org/abs/2603.22056) | arXiv 2026, ICASSP 2026 | Focuses on mismatched K/Q distributions across models with different tokenizers. The direct lesson is that interface alignment should target query-key geometry, not just output-token agreement. |
| [CoT2Align: Cross-Chain of Thought Distillation via Optimal Transport Alignment for Language Models with Different Tokenizers](https://arxiv.org/abs/2502.16806) | arXiv 2025 | Extends OT from token-level alignment to sequence-level and layer-wise alignment under tokenizer mismatch. That is very close to the LatentWire problem: bridge on relational structure, not raw index matching. |
| [Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamic Mapping](https://arxiv.org/abs/2502.11104) | arXiv 2025 | Uses context-conditional vocabulary mapping rather than a fixed remap. If LatentWire keeps a discrete or codebook-style interface, the mapping should likely depend on local context and route. |

## Why It Matters For LatentWire

- The recent Transformer-gauge papers make the failure mode sharper: a dense bridge can look bad simply because it is trying to match coordinates that are only defined up to head permutation and per-head linear gauge. That makes raw-space regression a weak default.
- The canonicalization papers suggest the right order of operations is often `canonicalize -> align/transport -> compress`, not `compress first and hope the bridge learns the symmetry away`.
- The quantization papers converge on one point: equal treatment of all coordinates is wasteful. Good compressors protect a small subset of directions, layers, or channels chosen by geometry or loss sensitivity. LatentWire should allocate interface bytes the same way.
- CommVQ and NSNQuant imply that preconditioning is part of the method, not mere implementation detail. A good interface basis may be one whose statistics are normalized and whose codebooks commute with known symmetries such as RoPE-style rotations.
- The cross-tokenizer KD papers are useful because they attack the same structural problem under a different name: source and target spaces differ in tokenization, sequence length, and attention geometry, so the bridge should align relational structure and K/Q statistics rather than only token ids or logits.
- The newest gauge-invariant diagnostics suggest a stricter evaluation standard: if a bridge win disappears after gauge-fixing or quotient-aware matching, the original result was probably measuring parameterization luck rather than real communication.

## Concrete Ablations / Diagnostics

1. `gauge_fix_then_bridge`: canonicalize each source/target head into an orthonormal value basis with balanced Q/K scaling before any transport; compare against the raw bridge at matched bytes and params.
2. `quotient_match_after_fix`: after gauge-fixing, solve the residual head permutation with Hungarian or OT matching; log residual transport cost, route entropy, and answer flips.
3. `protected_singular_channels`: build each routed message in an SVD basis and allocate high precision only to top singular directions or flip-salient channels; compare against uniform per-coordinate precision.
4. `interaction_aware_budgeter`: replace independent layer/route saliency with an IMPQ-style interaction model; solve a small knapsack/BQP over route atoms, layers, or channel groups under a byte budget.
5. `commutant_respecting_codebook`: compare plain VQ packets against Hadamard-normalized and RoPE-commuting codebooks for K/V-style interface messages.
6. `adaptive_canonicalizer`: test a small input-conditional canonicalizer chosen to maximize target confidence or verifier score before routing; compare with fixed SVD and fixed orthogonal gauges.
7. `gauge_free_metrics`: report CKA together with gauge-fixed Procrustes residual, OT cost, and a residual-permutation invariant such as a bispectral or sorted-head diagnostic; explicitly look for cases where CKA is high but transfer is poor.
8. `kq_geometry_alignment`: add a DSKD-style K/Q matching loss and a CoT2Align-style sequence/layer OT loss, then compare against plain latent MSE or logit-only supervision.

## Strongest Near-Term Experiments

- Start with `gauge_fix_then_bridge` plus `quotient_match_after_fix`; if this alone helps, the main blocker is mis-gauged coordinates rather than missing capacity.
- Pair `protected_singular_channels` with `interaction_aware_budgeter`; this directly tests whether LatentWire should spend bytes on compression-critical directions instead of semantically pretty ones.
- Run `commutant_respecting_codebook` only after a canonicalization pass; otherwise codebook quality and basis quality are confounded.
- Use `gauge_free_metrics` as a hard gate for future bridge changes; do not trust CKA-only improvements.
- If tokenizer or model-family mismatch remains a blocker, prioritize `kq_geometry_alignment` over more bridge depth.
