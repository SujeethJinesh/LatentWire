# Quantization / Compression Connector References for LatentWire

Web check: 2026-04-21. This memo prioritizes 2025-2026 primary sources that map directly onto LatentWire's AWQ / EXL2-inspired search: mixed-bit allocation, rate-distortion, KV-cache compression, and compression-as-communication math.

## Sources

| Date | Source | Why it matters for LatentWire |
|---|---|---|
| 2025-05-05 | [Radio: Rate-Distortion Optimization for Large Language Model Compression](https://arxiv.org/abs/2505.03031) / [OpenReview](https://openreview.net/forum?id=ifnxXCCEiM) | Strongest math-side anchor in this pass. It frames LLM compression directly as a rate-distortion problem, which is a better abstraction than ad hoc bit heuristics. |
| 2025-05-18 | [KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](https://arxiv.org/abs/2506.08018) | Importance-aware KV allocation with explicit gradient signals. Useful when LatentWire needs to decide where to spend bridge budget, not just how much budget exists. |
| 2025-02-21 | [SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention Computation](https://arxiv.org/abs/2502.15304) / [OpenReview](https://openreview.net/forum?id=lPSiw5ONHA) | Very asymmetric K-cache compression with channel-aware structure. Good inspiration for separate K/V policies and protected subspaces in cross-model transport. |
| 2025-07-24 | [Squeeze10-LLM: Squeezing LLMs' Weights by 10 Times via a Staged Mixed-Precision Quantization Method](https://arxiv.org/abs/2507.18073) | Staged mixed-precision search is a direct analogue for progressive bridge construction: coarse allocation first, then refinement on the hard residuals. |
| 2025-09-15 | [AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models](https://arxiv.org/abs/2509.12019) / [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1799/) | Shows that bit allocation can be treated as an automated search problem rather than a fixed heuristic. Useful if LatentWire learns a budget allocator for heads/channels/routes. |
| 2025-09-18 | [Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment](https://openreview.net/forum?id=l4F50jpiVH) | Closest recent analogue to EXL2-style mixed-bit allocation. The important idea is fractional/continuous budgeting under a distortion bound, not just integer bit choices. |
| 2025-10-05 | [PatternKV: Flattening KV Representation Expands Quantization Headroom](https://arxiv.org/abs/2510.05176) | Reparameterization before compression matters. This maps cleanly onto LatentWire's gauge-fix / canonicalize-then-bridge branch. |
| 2025-10-06 | [KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction](https://arxiv.org/abs/2510.05373) | Strong reminder that rotation plus a lightweight correction term can outperform naive direct quantization. This is a good template for bridge-side repair. |
| 2025-09-20 | [VQKV: High-Fidelity and High-Ratio Cache Compression via Vector-Quantization](https://openreview.net/forum?id=YyxvRDh4d4) | Codebook-style KV compression is the closest compression analogue to a learned latent bridge. Useful if LatentWire wants discrete bridge atoms instead of only dense projectors. |
| 2026-04-06 | [Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs](https://arxiv.org/abs/2604.04722) | Token-importance-based adaptive bit allocation is exactly the sort of runtime policy LatentWire needs if bridge budget must vary across tokens or repair rounds. |

## What To Borrow

- Treat bridge design as a rate-distortion optimization problem, not a pure projection problem.
- Split the allocation problem by subspace: K versus V, head versus head, and token versus token.
- Prefer staged search: canonicalize or rotate first, then allocate bits or routes to the residual error.
- Make the budget controller explicit. A learned or greedy allocator is more interpretable than a hidden all-or-nothing frontier rule.
- Prefer codebook or vector-quantized latent bridges when the goal is communication rather than raw regression.
- Keep the calibration target fixed: same prompt, same parser, same budget accounting, same sample ids.

## Concrete LatentWire Ablations

1. Replace the current frontier rule with a small rate-distortion controller and compare the full frontier curve, not just one operating point.
2. Test separate K and V budgets, including asymmetric protected-channel policies.
3. Compare `rotate/canonicalize -> compress` against `compress -> repair` on the same held-out-family toy.
4. Replace the dense bridge with a vector-quantized codebook bridge and measure whether the bridge becomes more interpretable.
5. Add a token-importance bit allocator that spends more budget on uncertain tokens and less on stable ones.
6. Run a staged mixed-precision bridge: coarse shared basis first, then residual refinement, then final decode.
7. Log `rate`, `distortion`, `bytes_transmitted`, `route_entropy`, `gauge_residual`, `head_match_accuracy`, and `repair_accept_rate` together so the math and the telemetry stay aligned.
8. Stress each ablation under held-out-family and tokenizer-mismatch settings to see whether gains survive transport noise rather than just fitting noise.
