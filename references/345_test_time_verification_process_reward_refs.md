# Test-Time Verification, Process Reward, and Repair References for LatentWire

Web check: 2026-04-21. Primary sources only.

The recurring pattern across this literature is not "sample more" in the abstract. It is:

1. spend extra test-time compute only when uncertainty warrants it,
2. score intermediate steps instead of only the final answer,
3. and keep verification separate from repair so we can tell which part actually helped.

## Reference Map

| Source | Primary link | Core idea | LatentWire ablation | Telemetry fields | Claim risk |
|---|---|---|---|---|---|
| **Self-Consistency Improves Chain of Thought Reasoning in Language Models** | [arXiv](https://arxiv.org/abs/2203.11171) | Sample multiple reasoning paths and marginalize them to a final answer instead of trusting one greedy trace. | Compare fixed-N route sampling against one best route, and against majority-vote selection over route candidates. | `candidate_count`, `route_diversity`, `answer_entropy`, `vote_margin`, `latency_ms` | Gains may come from extra samples alone, not from any verifier or repair mechanism. |
| **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** | [arXiv](https://arxiv.org/abs/2305.10601) | Search over coherent thoughts with self-evaluation and backtracking rather than one left-to-right decode. | Gate route branching on verifier uncertainty; compare greedy routing, shallow beam search, and limited backtracking. | `branch_factor`, `search_depth`, `backtrack_count`, `node_score`, `best_path_margin` | Search gains are easy to overclaim if the compute overhead is not counted explicitly. |
| **Let's Verify Step by Step** | [arXiv](https://arxiv.org/abs/2305.20050) | Process supervision outperforms outcome supervision for reasoning, and PRM800K makes step-level labels available. | Train or evaluate a route verifier on step labels vs final-answer labels; compare stepwise and outcome-only route scoring. | `step_label_source`, `step_accuracy`, `final_vs_step_gap`, `false_accept_rate`, `repair_help_rate` | Human step labels may not transfer cleanly to LatentWire telemetry or route traces. |
| **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning** | [arXiv](https://arxiv.org/abs/2410.08146) | A useful process reward should measure progress, i.e. whether a step increases the chance of a correct final answer. | Score route edits by estimated progress gain instead of raw correctness; compare progress-based vs outcome-based verifier ranking. | `pre_step_prob`, `post_step_prob`, `advantage`, `step_delta`, `search_efficiency` | A weak prover can invert the signal and make a bad step look useful. |
| **Generative Verifiers: Reward Modeling as Next-Token Prediction** | [arXiv](https://arxiv.org/abs/2408.15240) | Train the verifier as a generator so it can reason, emit critiques, and use majority voting over verifier rationales. | Replace a scalar route score with a generative verifier prompt over route telemetry and compare discriminative vs generative scoring. | `verifier_cot_len`, `rationale_vote_entropy`, `accept_rate`, `critique_consistency`, `token_budget` | A generative verifier can simply be a more expensive decoder unless cost is matched carefully. |
| **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** | [arXiv](https://arxiv.org/abs/2408.03314) | Allocate test-time compute adaptively per prompt; compute-optimal allocation beats naive best-of-N. | Difficulty-gated route expansion vs fixed-N routing; route more verifier passes only when uncertainty is high. | `prompt_difficulty`, `compute_alloc`, `extra_passes`, `budget_utilization`, `accuracy_per_token` | Adaptive allocation is only meaningful if the compute budget is measured consistently. |
| **The Lessons of Developing Process Reward Models in Mathematical Reasoning** | [arXiv](https://arxiv.org/abs/2501.07301) | PRM quality depends heavily on label synthesis and evaluation; MC labels can be weak and Best-of-N can be biased. | Compare MC self-labels, LLM-judge labels, and a small gold set for the verifier; check whether route selection changes. | `label_source`, `consensus_rate`, `boN_score`, `stepwise_auc`, `outcome_bias` | A PRM gain may reflect evaluation bias instead of better step verification. |
| **Process Reward Models That Think** | [arXiv](https://arxiv.org/abs/2504.16828) | Long-CoT PRMs can verify every step with a verbalized critique and need far fewer process labels. | Compare scalar route score vs long verification rationale vs critique-plus-repair on the same route pool. | `verifier_cot_len`, `first_error_step`, `repair_delta`, `selection_margin`, `token_cost` | The critic can become expensive enough that the test-time comparison is no longer fair. |
| **ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification** | [arXiv](https://arxiv.org/abs/2502.14565) | The model verifies its own reasoning and then revises it with confidence-aware decoding. | Add a self-verify-then-repair pass only for uncertain routes; compare revise-after-rerank vs rerank-only. | `self_verification_score`, `revision_rate`, `confidence`, `post_repair_accuracy`, `harm_rate` | Self-verification can collapse into self-justification unless prompts and decoding are controlled. |
| **Step-level Verifier-guided Hybrid Test-Time Scaling for Large Language Models** | [arXiv](https://arxiv.org/abs/2507.15512) | Fine-grained step-level self-refinement can be combined with parallel scaling at the step granularity. | Compare whole-route verifier control, step-level verifier control, and a hybrid branch-plus-repair policy. | `step_granularity`, `step_verifier_score`, `parallel_branch_count`, `first_correct_step`, `latency_ms` | Fine-grained control can look better only because it spends more compute. |
| **Budget-aware Test-time Scaling via Discriminative Verification** | [arXiv](https://arxiv.org/abs/2510.14913) | Cheap discriminative verification plus self-consistency can beat heavier generative verification under a fixed budget. | Use a cheap gate before any expensive verifier; only escalate ambiguous routes to the costly verifier. | `gate_precision`, `gate_recall`, `gen_calls`, `budget_saved`, `accuracy_per_token` | Hybrid gains vanish if the gate is not truly cheaper than the verifier it is protecting. |
| **Repair-R1: Better Test Before Repair** | [arXiv](https://arxiv.org/abs/2507.22853) | Generate discriminative tests before repair rather than validating only after the patch is proposed. | Compare test-before-repair, repair-then-verify, and repair-only policies on the same route pool. | `test_count`, `test_coverage`, `test_pass_rate`, `answer_change_rate`, `repair_success_rate` | In non-code tasks, synthetic tests may be too weak to drive meaningful repair. |

## Shared Telemetry Contract

If LatentWire is going to compare verifier and repair variants safely, log the same fields everywhere:

- `route_id`
- `candidate_rank`
- `verifier_score`
- `step_scores`
- `first_error_step`
- `repair_budget`
- `repair_latency_ms`
- `compute_passes`
- `candidate_count`
- `vote_margin`
- `route_entropy`
- `self_verification_score`
- `test_count`
- `test_coverage`
- `accuracy_per_token`
- `accuracy_per_byte`

## Highest-Priority LatentWire Ablations

1. Fixed-N self-consistency vs confidence-gated self-consistency.
2. Scalar outcome verifier vs step-level PRM vs generative verifier.
3. Rerank-only vs rerank-plus-targeted-repair on the same candidate pool.
4. Verify-then-repair vs repair-then-verify under the same token budget.
5. Cheap discriminative gate plus expensive verifier vs expensive verifier-only.

## Practical Readout

The paper-safe question is whether LatentWire still wins when bytes, latency, and verifier/repair passes are matched. If the answer is no, the result is still useful, but it is a budget allocation story, not a method story.
