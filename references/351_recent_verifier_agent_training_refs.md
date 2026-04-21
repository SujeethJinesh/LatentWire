# Recent Verifier, Communication, and Self-Correction References for LatentWire

Web check: 2026-04-21. Primary-source memo only. Use these papers to seed ablations and telemetry; do not treat any of them as evidence of a LatentWire gain until it holds on matched-budget held-out runs.

## Shared telemetry contract

Log these fields wherever they apply: `route_id`, `candidate_rank`, `verifier_score`, `verifier_margin`, `first_error_step`, `repair_changed_answer`, `repair_delta`, `token_budget`, `byte_budget`, `latency_ms`, `sample_seed`, `temperature`, `communication_rounds`, `message_bytes`, `protocol_id`, `agreement_rate`, `false_positive_rate`, `false_negative_rate`, `accuracy_per_token`, `accuracy_per_byte`.

## 1) Test-time verification and process reward models

- [ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://arxiv.org/abs/2412.06559) - Core idea: benchmark earliest-error localization on step-by-step math traces, not just final-answer correctness. LatentWire ablation: scalar route score vs step-localizer vs critique-plus-repair over the same candidate pool. Telemetry: `first_error_step`, `step_error_margin`, `verifier_confidence`, `repair_delta`, `token_budget`. Claim risk: `dev-smoke` until held-out error localization stays stable under matched compute.

- [xVerify: Efficient Answer Verifier for Reasoning Model Evaluations](https://arxiv.org/abs/2504.10481) - Core idea: answer-level equivalence judgment for long reasoning outputs, with explicit handling of final-answer extraction. LatentWire ablation: final-answer verifier vs step verifier vs hybrid route+answer verifier. Telemetry: `answer_equiv_score`, `parse_failure_rate`, `exact_match`, `selection_margin`, `false_negative_rate`. Claim risk: `dev-smoke` until verification is robust to formatting and extraction noise.

- [A*-Decoding: Token-Efficient Inference Scaling](https://arxiv.org/abs/2505.13672) - Core idea: structured search over partial solutions guided by process supervision, using budget where the frontier is promising. LatentWire ablation: best-of-N routes vs A*-style expansion vs verifier-guided pruning at matched budget. Telemetry: `frontier_size`, `expansion_count`, `PRM_passes`, `search_depth`, `token_budget`. Claim risk: `dev-smoke` until token savings survive the same accuracy target and do not just shift compute elsewhere.

- [Solve-Detect-Verify: Inference-Time Scaling with Flexible Generative Verifier](https://arxiv.org/abs/2505.11966) - Core idea: a generative verifier that detects when a solution is complete and triggers targeted verification instead of full-pass checking. LatentWire ablation: always-verify vs detect-and-verify vs no-verifier baseline. Telemetry: `trigger_point`, `verification_span`, `rollback_depth`, `verification_latency_ms`, `post_verify_accuracy`. Claim risk: `dev-smoke` until the trigger policy does not introduce hidden compute inflation.

- [V1: Unifying Generation and Self-Verification for Parallel Reasoners](https://openreview.net/forum?id=ZUFJQrZuRp) - Core idea: pairwise self-verification and tournament ranking outperform scalar scoring when many parallel candidates are available. LatentWire ablation: scalar ranking vs pairwise tournament vs uncertainty-gated pairwise ranking. Telemetry: `pairwise_comparisons`, `uncertainty_score`, `tournament_depth`, `selection_margin`, `agreement_rate`. Claim risk: `dev-smoke` until pairwise gains remain after candidate quality and diversity are controlled.

## 2) Agent communication protocols

- [LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration](https://arxiv.org/abs/2402.11550) - Core idea: leader/member collaboration with inter-member communication to resolve conflicts in long-context retrieval and QA. LatentWire ablation: isolated source-target messages vs conflict-aware message exchange vs shared-context summary messages. Telemetry: `message_count`, `conflict_rate`, `consensus_delta`, `bytes_sent`, `task_success`. Claim risk: `dev-smoke` until better collaboration is not just a byproduct of longer prompts.

- [A Scalable Communication Protocol for Networks of Large Language Models](https://arxiv.org/abs/2410.11905) - Core idea: Agora uses a meta-protocol that mixes standard routines, natural language, and LLM-written routines to sidestep the communication trilemma. LatentWire ablation: fixed-schema messages vs natural-language fallback vs routine-generated messages. Telemetry: `protocol_switches`, `message_entropy`, `task_completion_time`, `communication_overhead_ms`, `failure_recovery_rate`. Claim risk: `dev-smoke` until protocol gains persist after overhead is normalized.

- [ACPs: Agent Collaboration Protocols for the Internet of Agents](https://arxiv.org/abs/2505.13523) - Core idea: standardize registration, discovery, interaction, and tooling so heterogeneous agents can coordinate without ad hoc glue. LatentWire ablation: flat prompt-passing vs structured registration/discovery vs tool-carrying messages. Telemetry: `handshake_success`, `discoverability`, `tool_call_success`, `handoff_failure_rate`, `protocol_id`. Claim risk: `dev-smoke` until any wins survive across at least one non-trivial multi-agent task.

- [Which LLM Multi-Agent Protocol to Choose?](https://openreview.net/forum?id=lqNqKUG2dn) - Core idea: protocol choice itself changes utility, overhead, and resilience, so communication design should be benchmarked rather than hand-waved. LatentWire ablation: one protocol contract vs multiple protocol adapters vs oracle-selected protocol. Telemetry: `protocol_id`, `utility`, `overhead_ms`, `resilience_score`, `failure_mode`. Claim risk: `dev-smoke` until protocol choice is not confounded with task selection.

## 3) Debate and self-correction

- [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) - Core idea: online RL on self-generated correction traces avoids the distribution-mismatch and collapse problems of supervised self-correction. LatentWire ablation: direct answer rewrite vs self-correction RL policy vs no-correction control. Telemetry: `correction_turns`, `correction_delta`, `collapse_rate`, `on_policy_reward`, `test_time_gain`. Claim risk: `dev-smoke` until improvements hold when correction style is held constant.

- [DEBATE, TRAIN, EVOLVE: Self Evolution of Language Model Reasoning](https://arxiv.org/abs/2505.15734) - Core idea: debate traces plus Reflect-Critique-Refine can train a single model without ground truth. LatentWire ablation: single-agent critique vs two-agent debate vs debate-derived fine-tuning. Telemetry: `debate_rounds`, `critique_length`, `refine_help_rate`, `cross_domain_transfer`, `self_consistency_gain`. Claim risk: `dev-smoke` until debate traces improve out-of-domain reasoning, not just in-domain prompting.

- [Self-Correction Bench: Uncovering and Addressing the Self-Correction Blind Spot in Large Language Models](https://openreview.net/forum?id=7K1kXowjK1) - Core idea: many models can fix external errors but fail to activate the same capability for internal errors; a simple "Wait" prompt is a surprisingly strong baseline. LatentWire ablation: no-wait vs wait-prompt vs explicit error-injection repair. Telemetry: `blind_spot_rate`, `wait_trigger_rate`, `repair_success_rate`, `false_alarm_rate`, `error_type`. Claim risk: evaluation risk is high because error-injection setups can diverge from LatentWire failures.

- [When Debate Fails: Bias Reinforcement in Large Language Models](https://openreview.net/forum?id=c5bjw7hqix) - Core idea: debate can amplify bias instead of correcting it when agents lack perspective diversity and useful feedback. LatentWire ablation: homogeneous debaters vs heterogeneous debaters vs asymmetric critic/judge setup. Telemetry: `bias_shift`, `perspective_diversity`, `disagreement_rate`, `judge_flip_rate`, `harm_rate`. Claim risk: `dev-smoke` until debate helps on adversarial and bias-heavy prompts, not just clean math.

## 4) Causal and interpretable evaluation

- [Causal Evaluation of Language Models](https://arxiv.org/abs/2405.00622) - Core idea: CaLM evaluates association, intervention, counterfactuals, metric choice, and error types as a unified causal-reasoning benchmark. LatentWire ablation: factual-only eval vs causal-target eval vs counterfactual eval on the same verifier or router. Telemetry: `causal_target`, `adaptation_type`, `intervention_score`, `counterfactual_score`, `error_type`. Claim risk: `dev-smoke` until causal scores predict downstream behavior rather than benchmark-specific skill.

- [Compositional Causal Reasoning Evaluation in Language Models](https://arxiv.org/abs/2503.04556) - Core idea: test whether causal quantities compose correctly through graphs, not just whether the final answer is numerically plausible. LatentWire ablation: direct answer score vs causal-graph decomposed score vs graph-consistency verifier. Telemetry: `graph_depth`, `composition_error`, `path_consistency`, `calibration_gap`, `oracle_gap`. Claim risk: `dev-smoke` until the graph-based signal transfers to harder held-out tasks.

- [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207) - Core idea: latent coordinates and linear probes can expose structured geometry, so interpretability should be measured as geometry, not prose. LatentWire ablation: no-probe baseline vs geometry-aware routing vs probe-conditioned verification. Telemetry: `probe_r2`, `latent_linearity`, `route_separability`, `geometry_shift`, `probe_stability`. Claim risk: interpretability evidence only; do not count probe wins as task wins.

## Highest-priority LatentWire ablations

1. Scalar route scoring vs step-localization vs critique-plus-repair on the same candidate pool.
2. Best-of-N routing vs A*-style frontier expansion vs verifier-guided pruning at matched budget.
3. Fixed-schema communication vs natural-language fallback vs structured protocol handoff.
4. Direct self-rewrite vs RL-trained self-correction vs debate-derived fine-tuning.
5. Factual-only evaluation vs causal/counterfactual evaluation vs graph-consistency verification.
