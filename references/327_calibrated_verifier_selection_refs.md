# Calibrated Verifier and Selection References for LatentWire

Web check: 2026-04-21. This memo is additive to `320` and `325` and focuses on 2025-2026 methods that are most relevant to the current stochastic-route bottleneck: verifier collapse onto the target candidate, weak calibration across route sets, and the lack of an adaptive budget policy.

| Source | Primary link | Why it matters for LatentWire | Concrete LatentWire ablation |
|---|---|---|---|
| **CoRefine: Confidence-Guided Self-Refinement for Adaptive Test-Time Compute** | [arXiv](https://arxiv.org/abs/2602.08948) | Confidence is used as a control signal for whether to halt, re-examine, or try another path. That is a direct match for route expansion when the current selector is uncertain. | Gate 1/3/5 stochastic routes by target-alone entropy or route disagreement; compare fixed-N against confidence-triggered expansion under matched bytes. |
| **Adaptive Test-Time Reasoning via Reward-Guided Dual-Phase Search** | [arXiv](https://arxiv.org/abs/2509.25420) | Separating planning from execution suggests a cleaner LatentWire design: first choose a route family, then decide which route to decode or repair. | Split route selection into proposal and refinement phases; compare one-shot rerank against dual-phase search with the same candidate budget. |
| **REARANK: Reasoning Re-ranking Agent via Reinforcement Learning** | [arXiv](https://arxiv.org/abs/2505.20046) | Shows that listwise reranking improves when the selector reasons explicitly, rather than scoring candidates with a shallow scalar. This is useful when the verifier keeps collapsing to the target candidate. | Compare pointwise scoring, pairwise tournament, and listwise reasoning reranking over the same stochastic routes; add a small RL-style selector if prompt-only reranking saturates. |
| **SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling** | [arXiv](https://arxiv.org/abs/2501.19306) | Combines self-verification with self-correction, which is the most natural next step if reranking finds a near-miss route but not the exact answer. | Compare rerank-only, self-verify-only, and rerank-plus-repair; run the repair pass only on uncertain or near-tie candidates. |
| **Incentivizing LLMs to Self-Verify Their Answers** | [arXiv](https://arxiv.org/abs/2506.01369) | Internal self-verification reduces the mismatch between a generator and an external verifier. This is relevant because our current verifier is biased toward the target path and may not share the generator's route distribution. | Compare an external verifier, a self-verifier, and a hybrid verifier that sees the route trace plus the model's own confidence signal. |
| **Rank1: Test-Time Compute for Reranking in Information Retrieval** | [arXiv](https://arxiv.org/abs/2502.18418) | Makes reranking explicitly test-time-compute aware and explainable. The route-selection analogue is to rank candidate routes with a rationale instead of a single scalar. | Turn each stochastic route into a listwise item with rationale text and compare scalar scoring against listwise reasoning at fixed compute. |
| **Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning** | [arXiv](https://arxiv.org/abs/2601.13387) | Route quality may be better captured by a confidence trajectory than by a final scalar. That matters if the selector is missing early evidence of failure or late confidence spikes. | Log confidence per generation step and compare final-score selection against trajectory-based selection features such as monotonicity, drops, and late spikes. |

## What this means for LatentWire

- The current selector should not be treated as a fixed scalar verifier. It needs either confidence gating, listwise reasoning, or both.
- The most likely failure mode is label-order bias plus weak calibration, so the next selection experiments should randomize candidate order and log the position of the target candidate.
- If reranking still saturates, the next step should be a repair loop rather than a larger candidate pool.
- The paper should report both selection quality and compute efficiency. A selector that improves accuracy but doubles bytes is not a useful method claim.

## Highest-priority LatentWire ablations

1. **Randomized candidate order for the verifier.** Shuffle the target and seed route positions so we can measure whether the current collapse is an A-position artifact.
2. **Confidence-gated expansion.** Compare fixed 1/3/5-route sampling against uncertainty-triggered route expansion at matched average bytes.
3. **Pointwise vs pairwise vs listwise selection.** Keep the same candidate set and test whether the gain comes from better ranking structure or just from more compute.
4. **External verifier vs self-verifier vs hybrid.** Check whether the current target-model collapse is a distribution-mismatch problem rather than a candidate-quality problem.
5. **Rerank-only vs rerank-plus-repair.** For near-tie candidates, let the top route undergo one repair pass and measure whether correction recovers oracle-correct routes that pure reranking misses.
