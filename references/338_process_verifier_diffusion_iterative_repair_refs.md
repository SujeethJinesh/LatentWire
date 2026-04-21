# LatentWire Ablation Backlog: Process Verifiers, Repair, and Test-Time Refinement

Web check: 2026-04-21. This is a paper-safe note, not a claim list.

## Claim policy

- `Dev-smoke` means toy bridges, GSM30 slices, calibration splits, prompt-shape checks, and any run used to validate telemetry plumbing or budget accounting.
- `Held-out` means a frozen eval slice, fixed byte/token budget, identical serialization contract, and no hidden oracle routing or prompt changes between compared methods.
- Do not promote an ablation to a paper claim unless it is repeated on held-out data and wins under the same cost budget.

## Sources worth mining

- [ThinkPRM: Process Reward Models That Think](https://arxiv.org/abs/2504.16828). Core mechanism: a long-CoT verifier that generates step-wise verification rationales instead of a bare scalar. Why it helps cross-model communication: it exposes the first suspicious step, which is directly reusable as an interpretable repair cue across source and target models. Concrete ablation: scalar route score vs generated critique vs critique-plus-repair under the same decode budget. Telemetry fields: `verifier_cot_len`, `verifier_confidence`, `first_error_step`, `repair_help_rate`, `repair_harm_rate`, `pre_answer`, `post_answer`, `token_budget`. Claim risk: `dev-smoke` only until the same repair flow wins on held-out GSM70 / SVAMP with matched budget.

- [GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning](https://arxiv.org/abs/2504.00891). Core mechanism: a generative PRM that verifies each step with explicit reasoning and code checks. Why it helps cross-model communication: it turns verification into an active collaborator rather than a passive score, which matches LatentWire's repair-and-reselect workflow. Concrete ablation: generative step verifier vs discriminative step score vs no verifier, then feed the generated critique into target-side repair. Telemetry fields: `step_judgment`, `critique_length`, `code_check_pass`, `repair_delta`, `selection_margin`, `post_repair_accuracy`. Claim risk: `dev-smoke` until the same held-out budget envelope shows a Pareto win.

- [General Purpose Verification for Chain of Thought Prompting](https://arxiv.org/abs/2405.00204). Core mechanism: validate relevance, mathematical accuracy, and logical consistency, plus a perplexity-based verifier. Why it helps cross-model communication: those three axes map cleanly onto source-trace failures versus bridge failures versus target repair failures. Concrete ablation: three-axis verifier scores vs one scalar verifier vs perplexity-only ranking. Telemetry fields: `relevance_score`, `math_score`, `logic_score`, `perplexity`, `decision_margin`, `route_entropy`, `flip_rate`. Claim risk: `dev-smoke` unless the axis decomposition improves held-out answer accuracy under a fixed budget.

- [GM-PRM: A Generative Multimodal Process Reward Model for Multimodal Mathematical Reasoning](https://arxiv.org/abs/2508.04088). Core mechanism: a PRM that both critiques and corrects the first erroneous step. Why it helps cross-model communication: it is the closest published template for using a verifier as an active repair collaborator rather than a gatekeeper. Concrete ablation: repair prompt from the first-error critique vs full-trace repair vs no repair. Telemetry fields: `first_error_index`, `step_intent`, `visual_or_symbolic_alignment` where relevant, `repair_help_rate`, `repair_harm_rate`, `oracle_gap`. Claim risk: `dev-smoke` only unless the same repair contract survives held-out reasoning tasks.

- [Repair-R1: Better Test Before Repair](https://arxiv.org/abs/2507.22853). Core mechanism: generate discriminative tests before attempting repair. Why it helps cross-model communication: LatentWire can do the same thing by asking the target model to generate counterchecks against the selected route before it edits the final answer. Concrete ablation: repair-only vs test-before-repair vs test+repair under the same token budget. Telemetry fields: `test_count`, `test_coverage`, `test_pass_rate`, `repair_delta`, `answer_change_rate`, `final_correctness`. Claim risk: `dev-smoke` until test generation and repair jointly improve held-out accuracy at fixed budget.

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651). Core mechanism: generate, critique, and iteratively refine with the same model. Why it helps cross-model communication: it is the simplest template for asking the target model to repair its own imported reasoning instead of treating the bridge output as final. Concrete ablation: one-shot target answer vs one self-refine pass vs two-pass self-refine after source-route selection. Telemetry fields: `refine_round`, `critic_text_len`, `pre_post_agreement`, `pre_post_correctness`, `token_cost`, `stability`. Claim risk: `dev-smoke` unless the extra pass beats a matched-compute one-shot baseline.

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). Core mechanism: verbal feedback is stored and reused as episodic memory for later attempts. Why it helps cross-model communication: LatentWire can treat repair notes and verifier critiques as reusable bridge memory, not just transient text. Concrete ablation: discard repair notes vs reuse them across salts or downstream questions from the same source prompt. Telemetry fields: `memory_hit_rate`, `reflection_length`, `reuse_accuracy`, `episode_id`, `retry_count`, `help_rate`, `harm_rate`. Claim risk: `dev-smoke` until memory reuse helps on held-out prompts with the same cost envelope.

- [Large Language Models have Intrinsic Self-Correction](https://arxiv.org/abs/2406.15673). Core mechanism: self-correction works best under zero temperature and unbiased prompts. Why it helps cross-model communication: it gives a concrete prompt-control hypothesis for why some repairs help and others hallucinate. Concrete ablation: zero-temp repair vs sampled repair, and fair repair prompt vs answer-leading repair prompt. Telemetry fields: `temperature`, `prompt_bias_flags`, `repair_change_rate`, `repair_help_rate`, `repair_harm_rate`, `post_answer_confidence`. Claim risk: `dev-smoke` until prompt fairness and temperature effects hold on held-out data.

- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). Core mechanism: allocate test-time compute adaptively per prompt rather than uniformly. Why it helps cross-model communication: LatentWire should spend more repair budget on uncertain or high-disagreement routes, not on all routes equally. Concrete ablation: fixed repair budget vs confidence-adaptive repair budget vs compute-optimal prompt routing. Telemetry fields: `prompt_difficulty`, `repair_budget_used`, `selection_confidence`, `compute_alloc_ratio`, `accuracy_per_token`, `accuracy_per_byte`. Claim risk: `dev-smoke` until adaptive allocation wins under matched compute on a frozen split.

- [Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning](https://arxiv.org/abs/2502.00271). Core mechanism: verifier-guided search can degrade at larger sample sizes because imperfect verifiers prune valid paths. Why it helps cross-model communication: it is the warning label for over-trusting a route selector or repair critic. Concrete ablation: compare shallow repair, deep repair, and verifier-light repair to see when the bridge starts to prune the right route. Telemetry fields: `verifier_prune_rate`, `valid_path_loss`, `sample_size`, `oracle_gap`, `search_depth`, `repair_harm_rate`. Claim risk: `dev-smoke` unless the selected repair policy stays stable as search width increases.

- [Step-level Verifier-guided Hybrid Test-Time Scaling for Large Language Models](https://arxiv.org/abs/2507.15512). Core mechanism: fine-grained, step-level self-refinement combined with parallel test-time scaling. Why it helps cross-model communication: LatentWire can split repair into micro-steps and log which step actually changes the answer. Concrete ablation: single-shot repair vs step-level repair vs step-level plus parallel candidates. Telemetry fields: `step_granularity`, `step_verifier_score`, `parallel_branch_count`, `first_correct_step`, `post_step_accuracy`, `latency_ms`. Claim risk: `dev-smoke` until finer-grained repair improves held-out accuracy without inflating cost.

- [Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564). Core mechanism: iterative denoising over a continuous representation with geometric structure. Why it helps cross-model communication: it suggests a latent-space view of repair, where the bridge can iteratively denoise an intermediate representation before final decoding. Concrete ablation: direct decode vs one latent denoise pass vs multi-pass denoise before target generation. Telemetry fields: `latent_denoise_steps`, `latent_norm_change`, `decode_entropy`, `repair_delta`, `token_cost`, `final_correctness`. Claim risk: `dev-smoke` until the latent refinement path beats the matched-budget text-only repair path.

- [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://arxiv.org/abs/2512.13586). Core mechanism: slot-level plan-and-infill that recovers KV-cache reuse while keeping iterative refinement. Why it helps cross-model communication: it offers a direct design pattern for structured repair slots inside LatentWire, rather than free-form unstructured rewrites. Concrete ablation: free-form repair text vs slot-based repair plan vs slot-based plan-and-infill. Telemetry fields: `slot_count`, `plan_length`, `infill_accuracy`, `kv_reuse_fraction`, `repair_latency_ms`, `oracle_gap`. Claim risk: `dev-smoke` unless the structured repair flow wins on held-out data under the same byte budget.

## Telemetry contract

If a LatentWire repair ablation is meant to be useful later, it should log at least:

- `route_id`
- `candidate_rank`
- `verifier_score`
- `first_error_step`
- `pre_answer`
- `post_answer`
- `oracle_answer`
- `repair_changed_answer`
- `repair_help_rate`
- `repair_harm_rate`
- `repair_token_budget`
- `bytes_moved`
- `prompt_chars`
- `temperature`
- `sample_seed`

## Highest-priority moves

1. Extend the held-out GSM70 / SVAMP process-repair win with target self-repair,
   target-protection, and matched token/byte/latency budgets.
2. Compare scalar verification, step-level verification, and verifier-generated
   repair on the same bridge pool.
3. Add test-before-repair and step-level refinement variants, because they give
   the cleanest interpretability story.
4. Report accuracy-per-token and accuracy-per-byte alongside raw accuracy.
5. Keep every gain attributable to selection, verification, repair, or budget
   allocation rather than hidden compute inflation.
