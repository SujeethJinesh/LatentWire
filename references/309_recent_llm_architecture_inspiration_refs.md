# 309 Recent LLM Architecture Inspiration for LatentWire

Primary-source notes on 2025-2026 work that is plausibly useful for LatentWire. The goal here is not to copy these methods directly, but to extract ablation ideas and telemetry that help us diagnose cross-model latent communication.

## Short read

- The strongest recurring pattern is **interface specialization**: many recent systems get better by separating routing from content, or planning from rendering, or discrete from continuous representations.
- For LatentWire, the most actionable next step is to test whether our bridge improves when we **decouple K-style routing from V-style fidelity**, rather than treating cache compression as one uniform budget.
- The second theme is **token/vocab mismatch**. Recent tokenizer work suggests that vocabulary alignment is not cosmetic; it changes transfer quality, compression, and interpretability.
- The third theme is **iterative refinement**. Diffusion-style latent revision provides a natural control experiment for whether one-shot latent transport is the real blocker.

## Recent papers worth stealing from

| Theme | Paper | Primary link | Why it matters for LatentWire | Concrete ablation / telemetry |
|---|---|---|---|---|
| Multimodal routing instability | **RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer** (2025) | [arXiv](https://arxiv.org/abs/2506.11465), [code](https://github.com/GeWu-Lab/RollingQ_ICML2025) | Query rotation is being used to break modality collapse and redistribute attention. This is directly relevant to our rotalign family. | Test query rotation vs. value rotation vs. paired rotation; log key distribution gap, attention entropy, collapse rate, and source-target asymmetry. |
| Vocabulary alignment / transfer | **TokAlign: Efficient Vocabulary Adaptation via Token Alignment** (2025) | [arXiv](https://arxiv.org/abs/2506.03523) | Token ID mapping and embedding rearrangement are a direct analogue of cross-model token bridge design. | Add a tokenizer/vocab bridge ablation: token-ID mapping, embedding permutation, and progressive fine-tune; log vocab overlap, compression ratio, and transfer recovery. |
| Vocabulary specialization | **AdaptBPE: From General Purpose to Specialized Tokenizers** (2026) | [arXiv](https://arxiv.org/abs/2601.21665), [code](https://github.com/vijini/Adapt-BPE.git) | Suggests that a generic tokenizer can be the bottleneck; a specialized token inventory may reduce communication cost. | Test adapted tokenizer vs. shared tokenizer vs. mixed tokenizer; log bytes/token, perplexity proxy, and downstream accuracy under matched byte budgets. |
| Graph/token bridging | **Graph Tokenization for Bridging Graphs and Transformers** (2026) | [arXiv](https://arxiv.org/abs/2603.11099), [code](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers) | Reversible serialization + BPE is a useful template for turning structured latent state into a reusable token stream. | Try a reversible latent serialization before bridging; log reconstruction error, round-trip fidelity, and token merging frequency. |
| Attention-scheme routing | **Mixture of Attention Schemes (MoAS)** (2025) | [arXiv](https://arxiv.org/abs/2512.20650) | Dynamic choice between MHA/GQA/MQA suggests that LatentWire may benefit from routing not just tokens but attention mode. | Add per-token attention-scheme routing; log router entropy, scheme usage, and quality/latency trade-offs. |
| Directional attention control | **Directional Routing in Transformers** (2026) | [arXiv](https://arxiv.org/abs/2603.14923) | Learned suppression directions are a clean way to test whether the bridge needs explicit orientation control. | Add suppression-direction ablations around the bridge; log orientation span, signed cosine drift, and gate saturation. |
| Diffusion-style latent revision | **VDLM: Variable Diffusion LMs via Robust Latent-to-Text Rendering** (2026) | [arXiv](https://arxiv.org/abs/2602.15870) | Separates semantic planning from rendering and uses iterative refinement in latent space. Good control for one-shot vs. iterative bridge transport. | Compare one-shot bridge vs. multi-step refinement loop; log stepwise convergence, edit distance to previous step, and quality plateauing. |
| Refinement control | **Progressive Refinement Regulation for Accelerating Diffusion Language Model Decoding** (2026) | [arXiv](https://arxiv.org/abs/2603.04514) | Trajectory-grounded refinement control is a good template for deciding which bridge states should be revisited. | Add token-level reentry / revisit policy for latent slots; log revisit counts, convergence speed, and uncertain-slot recovery. |
| Prompt-level iterative optimization | **Prompt Optimization Via Diffusion Language Models** (2026) | [arXiv](https://arxiv.org/abs/2602.18449) | Shows that diffusion can optimize prompts without touching the target model; this is relevant if LatentWire should learn to reshape prompts instead of only caches. | Compare cache transport vs. prompt reshaping vs. hybrid; log prompt delta size, answer stability, and transfer cost. |
| Symmetry-aware model transfer | **Update Your Transformer to the Latest Release: Re-Basin of Task Vectors** (2025) | [arXiv](https://arxiv.org/abs/2505.22697), [code](https://github.com/aimagelab/TransFusion) | Directly relevant to permutation/symmetry handling in latent alignment. | Add permutation-matched bridge initialization and test whether performance only appears after canonicalization; log permutation stability and head matching confidence. |
| Symmetry-breaking / merge protection | **Model Unmerging: Making Your Models Unmergeable for Secure Model Sharing** (2025) | [arXiv](https://arxiv.org/abs/2509.01548), [code](https://github.com/hetailang/Merge-Lock) | Useful as a negative control: if small invertible transforms can hide mergeability, our bridge needs to measure recoverability, not just output match. | Add a recoverability test under invertible transforms; log whether the bridge can undo QK/VO reparameterization. |

## Most actionable LatentWire directions

1. **Split K and V budgets explicitly.** Treat routing/selectivity and content fidelity as different levers; do not collapse them into one cache-retention scalar.
2. **Add a tokenizer/vocab bridge.** Test token-ID remapping, embedding rearrangement, and specialized subword inventories before more adapter depth.
3. **Run iterative refinement as a control.** If a multi-step latent repair loop helps, then the blocker is transport instability rather than capacity.
4. **Make symmetry explicit.** Compare orthogonal, permutation-matched, and low-rank residual corrections; log gauge drift rather than only task accuracy.
5. **Measure router health, not just quality.** Track router entropy, scheme usage, collapse rate, route/value overlap, and round-trip reconstruction error.

## Telemetry to add or keep

- `kv_route_entropy`
- `kv_value_entropy`
- `route_value_overlap_jaccard`
- `route_value_kl`
- `orientation_span_avg`
- `signed_cosine_drift`
- `tokenizer_overlap_ratio`
- `bytes_per_correct_answer`
- `refinement_step_count`
- `revisit_rate`

## Source links

- RollingQ: https://arxiv.org/abs/2506.11465 and https://github.com/GeWu-Lab/RollingQ_ICML2025
- TokAlign: https://arxiv.org/abs/2506.03523
- AdaptBPE: https://arxiv.org/abs/2601.21665 and https://github.com/vijini/Adapt-BPE.git
- Graph Tokenization: https://arxiv.org/abs/2603.11099 and https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers
- MoAS: https://arxiv.org/abs/2512.20650
- Directional Routing: https://arxiv.org/abs/2603.14923
- VDLM: https://arxiv.org/abs/2602.15870
- PRR: https://arxiv.org/abs/2603.04514
- Prompt Optimization via DLMs: https://arxiv.org/abs/2602.18449
- Task-vector re-basin: https://arxiv.org/abs/2505.22697 and https://github.com/aimagelab/TransFusion
- MergeLock / model unmerging: https://arxiv.org/abs/2509.01548 and https://github.com/hetailang/Merge-Lock
