# Process Reward, Repair, and Verifier References for LatentWire

Web check: 2026-04-21. This memo is scoped to the current verifier failure mode: the pairwise/listwise judge collapses to the target path, does not calibrate well across route sets, and needs a stronger process signal than a final scalar score.

| Source | Primary link | Why it matters for LatentWire | Concrete LatentWire ablation |
|---|---|---|---|
| **Process Reward Models That Think** | [arXiv](https://arxiv.org/abs/2504.16828) | Strong recent evidence that step-wise verification can outperform pure outcome scoring when the selector can see intermediate reasoning. | Replace the current verifier with a step-scoring PRM and compare step-only, outcome-only, and hybrid aggregation on the same stochastic routes. |
| **GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning** | [arXiv](https://arxiv.org/abs/2504.00891) | Suggests the verifier itself should reason over the trace instead of just scoring it. That is a better fit for route ranking than a shallow classifier. | Compare scalar verifier, generative verifier, and generative verifier with explicit route rationale on the same candidate set. |
| **Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification** | [arXiv](https://arxiv.org/abs/2506.04592) | Step-aware retrospective checks are a direct match for detecting where a route first goes wrong. | Add first-error localization and step-level formal checks, then compare them against final-answer-only scoring. |
| **StepProof: Step-by-step verification of natural language mathematical proofs** | [arXiv](https://arxiv.org/abs/2506.10558) | Shows that verification quality improves when the text is normalized into verifier-friendly steps. | Canonicalize each route into step slots before judging and compare against raw route text and compressed summaries. |
| **Hard2Verify: A Step-Level Verification Benchmark for Open-Ended Frontier Math** | [arXiv](https://arxiv.org/abs/2510.13744) | Gives a recent benchmark framing for the exact problem we have: step-level verification is much harder than it looks, and self-verification can fail even when the answer is right. | Report calibration, self-vs-external verifier agreement, and step-level error localization on LatentWire route traces. |
| **ProgCo: Program Helps Self-Correction of Large Language Models** | [arXiv](https://arxiv.org/abs/2501.01264) | A concrete answer-repair mechanism: use programmatic verification before or during correction instead of only reranking. | Compare rerank-only, rerank-plus-repair, and deterministic repair rules for near-miss routes with the same compute budget. |
| **Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models** | [arXiv](https://arxiv.org/abs/2503.02623) | Confidence is not just a score; it can be trained and calibrated. That directly targets the verifier collapse and poor route gating. | Add confidence-calibrated gating for 1/3/5-route expansion and sweep temperature scaling or score calibration on the verifier output. |
| **Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?** / **BracketRank** / **Multi-Agent Debate for LLM Judges with Adaptive Stability Detection** | [arXiv](https://arxiv.org/abs/2508.17536), [arXiv](https://arxiv.org/abs/2604.08834), [arXiv](https://arxiv.org/abs/2510.12697) | These papers map to the exact structural choice in LatentWire selection: debate, vote, or tournament-style elimination. They are the closest analogues to pairwise verifier reranking. | Compare randomized pairwise tournament, listwise ranking, and vote-based aggregation on the same candidate routes; sweep bracket depth and stability cutoffs. |

## What to log in LatentWire

- Candidate-order seed, left/right orientation, and whether the target route appeared in position A.
- Step-level verifier scores, calibration metrics, and the first step where a route diverges from the target answer.
- Repair deltas: whether the corrected route changes the final answer, the route score, or both.
- Confidence traces across the route, not just the final scalar, so we can later build interpretable gates.

## Highest-priority ablations

1. Randomize candidate order in the verifier and log the target position so we can rule out an A-position bias.
2. Compare step-level PRM scoring against final-answer-only scoring on the same route pool.
3. Add rerank-plus-repair for near-tie candidates and measure recovery from oracle-correct routes.
4. Compare calibrated confidence gating against fixed 1/3/5 route budgets at matched average bytes.
5. Run pairwise tournament, listwise rank, and vote aggregation under the same candidate budget and compare calibration, not just accuracy.
