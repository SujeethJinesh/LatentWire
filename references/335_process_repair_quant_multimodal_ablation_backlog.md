# LatentWire Ablation Backlog: Process Repair, Quantization, Multimodal Slots, Diffusion

Web check: 2026-04-21. This is a paper-safe backlog, not a claim list. It maps recent process-verifier / process-reward, quantization, multimodal adapter, and iterative-refinement ideas into concrete LatentWire ablations.

## Claim policy

- `Dev-smoke` means: toy tasks, GSM30 slices, calibration splits, prompt-shape checks, and any run used to validate telemetry plumbing or budget accounting.
- `Held-out` means: fixed protocol, disjoint eval slice, same budget as control, and no prompt / serialization changes between compared methods.
- Do not promote any ablation to a paper claim unless it is repeated on held-out data and wins under the same cost budget.

## 1) Process-verifier / process-reward family

Primary sources:

- [ThinkPRM: Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning](https://arxiv.org/abs/2504.00891)
- Optional comparator: [VRPRM](https://arxiv.org/abs/2508.03556) and [GM-PRM](https://arxiv.org/abs/2508.04088) as later multimodal extensions

Why it matters:

- LatentWire already shows that selector quality is not enough; the strongest live lane is now process repair. PRM-style verification turns the bridge into a step-aware audit + repair loop instead of a one-shot scorer.
- Generative PRMs matter because they can expose the first wrong step, not only a scalar score. That is directly usable for interpretable telemetry and for repair prompts.

Concrete LatentWire ablations:

- Verifier-only rerank of candidate routes before any repair.
- Verifier + repair loop: select route, ask target model to diagnose, then ask for a corrected final answer.
- Multi-pass process repair: one repair pass vs two passes vs early-stop on confidence stabilization.
- First-error localization: ask the verifier to name the first failing step, then compare against free-form repair.
- Reward-guided search under a fixed token budget.

Telemetry fields:

- `pre_repair_accuracy`
- `post_repair_accuracy`
- `changed_answer_rate`
- `help_rate`
- `harm_rate`
- `oracle_route_accuracy`
- `verifier_confidence`
- `first_error_index`
- `step_count`
- `prompt_chars`
- `repair_token_budget`
- `repair_edit_distance`

Paper-safe note:

- `Dev-smoke` can use GSM30 and toy bridges to validate prompt structure and repair diagnostics.
- `Held-out` should use GSM70, SVAMP, or another frozen split with the exact same selection / repair contract and byte budget.

## 2) Quantization / KV compression family

Primary sources:

- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [AQLM](https://arxiv.org/abs/2401.06118)
- [ExLlamaV2 / EXL2 docs](https://docs.mistral.ai/cookbooks/concept-deep-dive-quantization-methods-exl2) and [ExLlamaV2 repository](https://github.com/turboderp/exllamav2)

Why it matters:

- These methods are useful design priors for LatentWire even when the end goal is not compression itself: they show how to allocate a fixed budget to the most sensitive channels, layers, or tokens.
- The quantization literature also gives a clean vocabulary for budget-aware telemetry: bytes, bits-per-weight, layer sensitivity, activation outliers, and calibration-set overfitting.

Concrete LatentWire ablations:

- Importance-aware budget allocation across source layers, target layers, and route atoms.
- Mixed-precision latent bridge: keep a small number of high-sensitivity channels at higher precision.
- KV-cache compression as a communication bottleneck, not only as an efficiency trick.
- Budget sweep: fixed bytes vs fixed tokens vs fixed accuracy target.
- Calibration-set sensitivity: compare selection / repair quality across small vs larger calibration pools.

Telemetry fields:

- `avg_bits_per_weight`
- `layer_bit_allocation`
- `retained_kv_fraction`
- `bytes_moved`
- `latency_ms`
- `accuracy`
- `calibration_loss`
- `quantization_error_by_layer`
- `selector_entropy`
- `top1_mass`
- `source_target_asymmetry`
- `token_per_byte`

Paper-safe note:

- Quantization-inspired ablations are `Dev-smoke` until they are repeated on held-out tasks at matched cost.
- Do not claim a method win from compression alone unless the same budgeted protocol beats the control on held-out data.

## 3) Multimodal adapter / Perceiver / Q-Former family

Primary sources:

- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [Flamingo](https://arxiv.org/abs/2204.14198)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)

Why it matters:

- These models solve the same structural problem LatentWire has: compress an upstream representation into a small number of learned query slots without losing the useful signal.
- They are the best source for slot-based, interpretable bridge designs when direct token transport is too dense or too brittle.

Concrete LatentWire ablations:

- Fixed query bank vs learned query bank.
- Coarse-to-fine slot refinement vs flat one-shot projection.
- Target-conditioned query slots vs source-conditioned slots.
- Locality-preserving slot pooling vs global pooling.
- Frozen source / target with trainable bridge only.

Telemetry fields:

- `query_slot_occupancy`
- `dead_slot_rate`
- `slot_entropy`
- `source_token_coverage`
- `slot_to_token_alignment`
- `attention_sparsity`
- `per_slot_contribution`
- `byte_budget_per_slot`
- `layerwise_routing`
- `route_overlap_with_runtime_heads`

Paper-safe note:

- Slot adapters are `Dev-smoke` if trained or tuned on calibration data only.
- A held-out claim needs the same slot budget, same prompt contract, and no hidden oracle routing.

## 4) Diffusion / iterative refinement family

Primary sources:

- [Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)
- [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://arxiv.org/abs/2512.13586)
- Optional hybrid reference: [MADFormer](https://arxiv.org/abs/2506.07999)

Why it matters:

- Process repair already suggests that a one-shot bridge is often too brittle. Iterative refinement is the natural next step if the target model can fix partial outputs better than it can select them.
- Diffusion-style planning / denoising gives a principled way to keep uncertainty around instead of collapsing early to a wrong answer.

Concrete LatentWire ablations:

- One-pass repair vs `k`-step denoise / repair.
- Blockwise refinement for long reasoning traces.
- Early-stop on convergence of answer and confidence.
- Preserve low-confidence spans as belief states before final emission.
- Repair-after-selection vs refinement-before-selection.

Telemetry fields:

- `step_accuracy_curve`
- `answer_change_rate_by_step`
- `confidence_trajectory`
- `edit_distance_by_step`
- `convergence_step`
- `stop_step`
- `repair_help_rate`
- `repair_harm_rate`
- `budget_per_iteration`
- `token_uncertainty`

Paper-safe note:

- Iterative refinement is `Dev-smoke` until it survives held-out evaluation under the same total token budget as the one-pass control.
- The paper should report whether refinement helps because it fixes the first error, not merely because it spends more compute.

## Highest-priority LatentWire moves

1. Promote process repair from a dev-smoke lane to a held-out lane with fixed budgets and a strict final-answer contract.
2. Replace flat bridge projections with learned query-slot adapters, because the multimodal literature is the clearest template for a small, interpretable bottleneck.
3. Add budget-aware quantization / KV compression ablations so the paper can report accuracy-per-byte, not only raw accuracy.
4. Test a small iterative refinement loop after route selection, since the current strongest signal is that repair can improve selected routes without observed harm on the GSM30 smoke.
