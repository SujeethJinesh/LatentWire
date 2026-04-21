# LatentWire Reference Note: Recent Reasoning, Repair, and Multimodal Alignment

Web check: 2026-04-21. This is a paper-safe reference note, not a claim list. It is meant to seed ablations and telemetry, not to justify conclusions before held-out, matched-budget evaluation.

## Claim policy

- `Dev-smoke` means toy bridges, GSM30 slices, calibration splits, prompt-shape checks, or any run used to validate telemetry plumbing and budget accounting.
- `Held-out` means a frozen eval slice, fixed byte/token budget, identical serialization contract, and no hidden oracle routing or prompt changes between compared methods.
- Do not promote an ablation to a paper claim unless it is repeated on held-out data and wins under the same cost budget.
- For every gain, log both raw accuracy and efficiency-normalized metrics such as `accuracy_per_token` and `accuracy_per_byte`.

## 1) Process verifiers and test-time compute

- [ThinkPRM: Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
  - What to borrow: long-CoT verification instead of a bare scalar score; use the verifier as a critique generator that names the first suspicious step.
  - Concrete ablations: scalar route score vs generated critique vs critique-plus-repair; best-of-N route selection vs repair-guided selection; zero-temp vs sampled verifier.
  - Telemetry fields: `verifier_cot_len`, `verifier_confidence`, `first_error_step`, `route_entropy`, `repair_help_rate`, `repair_harm_rate`, `token_budget`.
  - Claim risk: `dev-smoke` until held-out GSM70 / SVAMP70 repeats under matched budget.

- [GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning](https://arxiv.org/abs/2504.00891)
  - What to borrow: generative step verification plus explicit reasoning before judgment; treat verification as an active collaborator.
  - Concrete ablations: discriminative step score vs generative critique vs no verifier; feed generated critique into target-side repair.
  - Telemetry fields: `step_judgment`, `critique_length`, `code_check_pass`, `selection_margin`, `repair_delta`, `post_repair_accuracy`.
  - Claim risk: `dev-smoke` until it is shown to improve held-out performance without hidden compute inflation.

- [Self-Enhanced Test-Time Scaling (SETS)](https://arxiv.org/abs/2501.19306)
  - What to borrow: sampling, self-verification, and self-correction as one test-time loop.
  - Concrete ablations: fixed repair budget vs confidence-adaptive repair budget vs self-correct-then-repair.
  - Telemetry fields: `prompt_difficulty`, `repair_budget_used`, `selection_confidence`, `compute_alloc_ratio`, `latency_ms`.
  - Claim risk: `dev-smoke` unless adaptive allocation wins under the same compute envelope.

- [ARIES: Stimulating Self-Refinement of Large Language Models by Iterative Preference Optimization](https://arxiv.org/abs/2502.05605)
  - What to borrow: iterative refinement loops that generate progressively better answers, then filter them into the next round.
  - Concrete ablations: one-pass repair vs multi-pass repair vs repair trajectories ranked by a simple rule-based selector.
  - Telemetry fields: `refine_round`, `trajectory_count`, `pre_post_agreement`, `pre_post_correctness`, `stability`.
  - Claim risk: `dev-smoke` until the extra pass beats a matched-compute one-shot baseline.

- [Large Language Models have Intrinsic Self-Correction](https://arxiv.org/abs/2406.15673)
  - What to borrow: zero-temperature and fair prompts matter for self-correction; prompt bias can dominate apparent gains.
  - Concrete ablations: zero-temp repair vs sampled repair; fair repair prompt vs answer-leading repair prompt.
  - Telemetry fields: `temperature`, `prompt_bias_flags`, `repair_change_rate`, `repair_help_rate`, `repair_harm_rate`.
  - Claim risk: `dev-smoke` until prompt fairness and temperature effects hold on held-out data.

- [Repair-R1: Better Test Before Repair](https://arxiv.org/abs/2507.22853)
  - What to borrow: generate discriminative tests before editing the final answer.
  - Concrete ablations: repair-only vs test-before-repair vs test+repair under the same token budget.
  - Telemetry fields: `test_count`, `test_coverage`, `test_pass_rate`, `answer_change_rate`, `final_correctness`.
  - Claim risk: `dev-smoke` until test generation and repair jointly improve held-out accuracy at fixed budget.

- [Step-level Verifier-guided Hybrid Test-Time Scaling](https://arxiv.org/abs/2507.15512)
  - What to borrow: fine-grained, step-level verification rather than a single sequence-level judgment.
  - Concrete ablations: single-shot repair vs step-level repair vs step-level plus parallel candidate branches.
  - Telemetry fields: `step_granularity`, `step_verifier_score`, `parallel_branch_count`, `first_correct_step`, `post_step_accuracy`.
  - Claim risk: `dev-smoke` until finer-grained repair improves held-out accuracy without inflating cost.

## 2) Multimodal projector and alignment methods

- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
  - What to borrow: dynamic resolution and a unified multimodal position scheme; do not force all inputs through one brittle fixed token layout.
  - Concrete ablations: fixed-size serialization vs dynamic-resolution serialization; route pooling by content complexity.
  - Telemetry fields: `input_resolution`, `token_count`, `serialization_shape`, `route_entropy`, `bytes_moved`.
  - Claim risk: `dev-smoke` unless any routing gain survives a frozen prompt contract and matched bytes.

- [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/abs/2408.03326)
  - What to borrow: transfer across image, multi-image, and video settings with one backbone.
  - Concrete ablations: single-route vs multi-route communication, and source/target swapping on multi-instance inputs.
  - Telemetry fields: `modality_count`, `route_switch_rate`, `cross_scenario_accuracy`, `repair_harm_rate`.
  - Claim risk: `dev-smoke` until the same mechanism helps across more than one held-out setting.

- [Libra: Building Decoupled Vision System on Large Language Models](https://arxiv.org/abs/2405.10140)
  - What to borrow: decouple inner-modal modeling from cross-modal interaction; use different attention behavior for within-modality vs bridge steps.
  - Concrete ablations: shared bridge vs decoupled inner/cross bridge; routed expert vs single bridge; bridge-only vs bridge-plus-repair.
  - Telemetry fields: `inner_modal_steps`, `cross_modal_steps`, `route_switch_rate`, `bridge_attention_mass`, `target_selection`.
  - Claim risk: `dev-smoke` unless the decoupling improves held-out accuracy at fixed byte budget.

- [AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Understanding](https://arxiv.org/abs/2502.01341)
  - What to borrow: explicit latent-space alignment instead of hoping a generic MLP connector will be enough.
  - Concrete ablations: standard MLP connector vs text-space weighted-average alignment vs bridge-side latent alignment regularizer.
  - Telemetry fields: `embedding_angle`, `latent_distance`, `alignment_loss`, `patch_entropy`, `post_repair_correctness`.
  - Claim risk: `dev-smoke` until alignment gains are shown on held-out data and not just calibration slices.

- [BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](https://arxiv.org/abs/2508.06895)
  - What to borrow: supervise the projector with refined internal embeddings rather than only final text loss.
  - Concrete ablations: final-answer-only training vs direct embedding supervision vs logit-distribution matching.
  - Telemetry fields: `projector_cosine`, `logit_kl`, `embedding_shift`, `patch_alignment_score`, `repair_delta`.
  - Claim risk: `dev-smoke` until direct supervision helps on a held-out split without just overfitting projector statistics.

- [Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models](https://arxiv.org/abs/2505.17316)
  - What to borrow: patch-level alignment and compression analysis; coarse alignment is often not enough.
  - Concrete ablations: patch-aligned training vs caption-loss-only training vs no alignment regularizer.
  - Telemetry fields: `patch_alignment_score`, `compression_ratio`, `semantic_coverage`, `route_entropy`, `oracle_gap`.
  - Claim risk: `dev-smoke` until the same signal improves answer accuracy on held-out tasks.

## 3) Diffusion and iterative refinement

- [Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)
  - What to borrow: iterative denoising over a continuous latent representation; think of repair as latent cleanup rather than only text rewrite.
  - Concrete ablations: direct decode vs one latent denoise pass vs multi-pass latent denoise before target generation.
  - Telemetry fields: `latent_denoise_steps`, `latent_norm_change`, `decode_entropy`, `repair_delta`, `final_correctness`.
  - Claim risk: `dev-smoke` until the latent path beats the matched-budget text-only repair path.

- [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://arxiv.org/abs/2512.13586)
  - What to borrow: slot-based plan-and-infill with explicit parallelism and better KV reuse.
  - Concrete ablations: free-form repair text vs slot-based repair plan vs slot-based plan-and-infill.
  - Telemetry fields: `slot_count`, `plan_length`, `infill_accuracy`, `kv_reuse_fraction`, `repair_latency_ms`.
  - Claim risk: `dev-smoke` until structured repair wins on held-out data under the same byte budget.

- [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)
  - What to borrow: blockwise latent thought tokens with adaptive test-time compute; this is a clean fit for route summaries.
  - Concrete ablations: scalar route summary vs block latent route summary vs latent route summary plus repair.
  - Telemetry fields: `latent_block_count`, `thought_token_budget`, `diversity_score`, `route_selection_margin`, `pre_post_correctness`.
  - Claim risk: `dev-smoke` until the latent route summary improves held-out reasoning under matched compute.

- [Soft-Masked Diffusion Language Models](https://arxiv.org/abs/2510.17206)
  - What to borrow: keep partial information when tokens are not revised, instead of discarding them completely.
  - Concrete ablations: hard mask vs soft mask vs repair-aware soft mask in bridge serialization.
  - Telemetry fields: `mask_retention_rate`, `partial_prior_strength`, `sequence_perplexity`, `repair_delta`.
  - Claim risk: `dev-smoke` until soft retention improves held-out generation quality under the same budget.

- [Finish First, Perfect Later: Test-Time Token-Level Cross-Validation for Diffusion Large Language Models](https://arxiv.org/abs/2510.05090)
  - What to borrow: accept a draft first, then cross-validate and remask weak tokens during refinement.
  - Concrete ablations: one-shot refinement vs cross-validated refinement vs cross-validated refinement plus route repair.
  - Telemetry fields: `accepted_token_fraction`, `remask_rate`, `cross_validation_score`, `final_accuracy`.
  - Claim risk: `dev-smoke` until iterative remasking helps more than a matched-budget non-remasking baseline.

- [Corrective Diffusion Language Models](https://arxiv.org/abs/2512.15596)
  - What to borrow: train or steer for error-aware confidence and targeted in-place correction.
  - Concrete ablations: standard denoise vs corrective denoise vs error-aware confidence-guided repair.
  - Telemetry fields: `error_confidence`, `token_revise_rate`, `preserve_correct_rate`, `correction_success_rate`.
  - Claim risk: `dev-smoke` until correction behavior is verified on held-out data and not just controlled settings.

## 4) Lateral architecture and representation ideas

- [NextLevelBERT: Masked Language Modeling with Higher-Level Representations for Long Documents](https://arxiv.org/abs/2402.17682)
  - What to borrow: operate on higher-level semantic chunks, not only token streams.
  - Concrete ablations: token-level bridge vs chunk-level bridge vs chunk-level bridge plus repair.
  - Telemetry fields: `chunk_size`, `semantic_chunk_entropy`, `cross_chunk_consistency`, `repair_delta`.
  - Claim risk: `dev-smoke` until chunking improves held-out accuracy without hiding errors in over-compressed summaries.

- [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207)
  - What to borrow: linear structure and interpretable coordinates in latent space; useful as a design and analysis lens for route geometry.
  - Concrete ablations: scalar score only vs scalar plus latent geometry probes vs geometry-aware route selection.
  - Telemetry fields: `probe_accuracy`, `latent_linearity`, `route_distance`, `selection_margin`.
  - Claim risk: `dev-smoke` because interpretability signals are not the same as downstream accuracy.

- [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://arxiv.org/abs/2401.12168)
  - What to borrow: human-aligned discretization and explicit geometric structure when the downstream task is spatial or symbolic.
  - Concrete ablations: raw numeric route metadata vs discretized/human-aligned metadata vs metadata plus repair.
  - Telemetry fields: `rounding_rule`, `human_alignment_score`, `numeric_error`, `repair_help_rate`.
  - Claim risk: `dev-smoke` until the structured metadata improves held-out symbolic reasoning rather than only making outputs look cleaner.

- [Gurnee & Tegmark, 2024](https://arxiv.org/abs/2310.02207)
  - What to borrow: linear probes for space/time-like structure as a sanity check for whether the bridge space is organized or just noisy.
  - Concrete ablations: bridge latent probes before repair vs after repair vs no repair.
  - Telemetry fields: `probe_r2`, `latent_curvature`, `route_separability`, `pre_post_probe_shift`.
  - Claim risk: interpretability evidence only; do not treat probe wins as task wins.

## 5) Telemetry contract for LatentWire

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
- `latency_ms`
- `accuracy_per_token`
- `accuracy_per_byte`

## 6) Highest-priority moves

1. Extend the held-out GSM70 / SVAMP process-repair win with target self-repair, target-protection, and matched token/byte/latency budgets.
2. Compare scalar verification, step-level verification, and verifier-generated repair on the same bridge pool.
3. Add test-before-repair and step-level refinement variants because they give the cleanest interpretability story.
4. Report accuracy-per-token and accuracy-per-byte alongside raw accuracy.
5. Keep every gain attributable to selection, verification, repair, or budget allocation rather than hidden compute inflation.
