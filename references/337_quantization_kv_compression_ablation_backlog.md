# LatentWire Ablation Backlog: Quantization, KV Compression, Outliers, and Budget Allocation

Web check: 2026-04-21. This is a paper-safe backlog, not a claim list. It maps quantization and KV-cache compression ideas into concrete LatentWire ablations for cross-model communication and interpretable telemetry.

## Claim policy

- `Dev-smoke` means: toy bridges, GSM30 slices, calibration splits, prompt-shape checks, and any run used to validate telemetry plumbing or budget accounting.
- `Held-out` means: frozen eval slice, fixed byte / token budget, identical serialization contract, and no hidden oracle routing or prompt changes between compared methods.
- Do not promote an ablation to a paper claim unless it is repeated on held-out data and wins under the same cost budget.

## 1) Weight quantization: EXL2 / GPTQ / AWQ / SmoothQuant

Primary sources:

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [ExLlamaV2 repository](https://github.com/turboderp-org/exllamav2) and [EXL2 docs](https://exllamav2.netlify.app/)

Why it matters:

- GPTQ gives the most direct sensitivity-weighted view of compression: preserve directions that matter most under approximate second-order information.
- SmoothQuant is the cleanest activation-outlier story: migrate difficulty from activations into weights so the downstream path is easier to quantize.
- AWQ is the clearest activation-aware weight selection story: keep the channels that are most important for preserving output under low-bit compression.
- EXL2 is not a separate theory paper, but it is a practical mixed-bit implementation template that mirrors the same design tension LatentWire has: allocate a fixed communication budget unevenly where it matters most.

Concrete LatentWire ablations:

- Mixed-bit latent bridge: keep high-sensitivity bridge channels in higher precision while compressing low-sensitivity channels more aggressively.
- GPTQ-style sensitivity weighting over bridge projections: rank bridge dimensions or route atoms by approximate Hessian / curvature proxy, then quantize or prune the least sensitive first.
- SmoothQuant-style activation smoothing before route selection: rescale high-outlier source / target activations and check whether selector entropy becomes less degenerate.
- AWQ-style channel protection: keep a small protected set of bridge channels / heads / slots that dominate the selected route score.
- EXL2-style bitrate sweep: compare one global bridge bitrate against heterogeneous per-layer / per-head / per-slot bitrates.
- Quantize source-side features, target-side features, or both, and record which side is more fragile.

Telemetry fields:

- `avg_bits_per_bridge_channel`
- `layer_bit_allocation`
- `protected_channel_fraction`
- `activation_outlier_mass`
- `quantization_error_by_layer`
- `selector_entropy`
- `route_entropy`
- `top1_mass`
- `pre_post_answer_agreement`
- `bytes_moved`
- `token_per_byte`

Paper-safe note:

- `Dev-smoke` can use GSM30 and toy bridges to validate quantization plumbing and sensitivity scoring.
- `Held-out` should only be claimed if the same mixed-bit contract beats the control on GSM70, SVAMP, or another frozen split at matched byte budget.

## 2) KV-cache compression: KVPress / KVzip / Quest / KV-Compress / VL-Cache

Primary sources:

- [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774)
- [KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head](https://arxiv.org/abs/2410.00161)
- [VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration](https://arxiv.org/abs/2410.23317)
- [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](https://arxiv.org/abs/2505.23416)
- [NVIDIA KVPress repository](https://github.com/NVIDIA/kvpress)

Why it matters:

- Quest is the best query-aware baseline: it turns KV selection into a current-query decision and therefore gives LatentWire a direct comparator for route-aware memory pressure.
- KV-Compress makes budget allocation explicit at the head level, which maps well to LatentWire if source / target routes have uneven layer or head sensitivity.
- VL-Cache shows how modality and sparsity interact with budget allocation; the same idea can be reused for source-vs-target asymmetry in cross-model communication.
- KVzip is the strongest query-agnostic comparator: if a LatentWire bridge needs to be reusable across prompts, it should be compared against a compression policy that tries to preserve general context rather than only the current query.
- KVPress is the tooling template for turning these ideas into reusable, benchmarkable compression policies.

Concrete LatentWire ablations:

- Query-aware route retention vs query-agnostic route retention.
- Per-head / per-layer route budgets instead of one global budget.
- Context reconstruction on compressed routes before target-side repair.
- Route reuse across multiple downstream questions from the same source prompt.
- Compress source-side context first, target-side context first, or both, and compare the asymmetry.
- Head-aware and layer-aware eviction of low-contribution route atoms.
- Fixed budget versus adaptive budget allocation based on selector confidence.

Telemetry fields:

- `retained_kv_fraction`
- `per_head_budget`
- `per_layer_budget`
- `reconstruction_loss`
- `query_reuse_accuracy`
- `route_overlap_with_query`
- `latency_ms`
- `bytes_moved`
- `attention_sparsity`
- `selected_token_coverage`

Paper-safe note:

- `Dev-smoke` can use GSM30 route pools, prompt perturbations, and toy multiple-query tests.
- `Held-out` needs frozen downstream queries and a matched memory budget; reuse across prompts is only a claim if accuracy survives that setting.

## 3) Activation outliers and robustness

Primary sources:

- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [GPTQ](https://arxiv.org/abs/2210.17323)

Why it matters:

- Activation outliers are the cleanest explanation for brittle route selection and brittle repair prompts.
- If LatentWire is collapsing onto a few extreme features, then a smoothing or protection mechanism should improve both accuracy and interpretability.

Concrete LatentWire ablations:

- Outlier-aware normalization before route scoring.
- Clip / smooth the top activation spikes and measure whether selector collapse decreases.
- Compare bridge quality with and without protected outlier channels.
- Sensitivity under prompt perturbation: if one outlier token dominates the route, small prompt changes should measurably change the route distribution.
- Compare route entropy before and after outlier handling to see whether the bridge becomes less brittle.

Telemetry fields:

- `outlier_token_fraction`
- `outlier_channel_fraction`
- `route_entropy_before`
- `route_entropy_after`
- `prompt_perturbation_sensitivity`
- `selector_flip_rate`
- `topk_outlier_mass`
- `answer_stability`

Paper-safe note:

- These are `Dev-smoke` until they are shown to improve held-out accuracy under the same budget.
- If outlier handling improves telemetry but not final answers, report it as interpretability evidence only, not as a method win.

## 4) Mixed precision and budget allocation

Primary sources:

- [AWQ](https://arxiv.org/abs/2306.00978)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [KV-Compress](https://arxiv.org/abs/2410.00161)
- [VL-Cache](https://arxiv.org/abs/2410.23317)
- [Quest](https://arxiv.org/abs/2406.10774)
- [KVzip](https://arxiv.org/abs/2505.23416)

Why it matters:

- LatentWire should not report only raw accuracy; it should report accuracy per byte, per token, and per repaired route.
- Mixed precision is the right lens for bridge design because the bridge is really a communication budget allocation problem.

Concrete LatentWire ablations:

- Same total byte budget, different distribution across source, bridge, and target.
- Same total token budget, different distribution across selection, compression, and repair.
- Compare fixed budget versus confidence-adaptive budget.
- Compare calibration-budget allocation with a uniform budget baseline.
- Measure whether a small set of protected high-precision route atoms dominates performance.
- Run an `accuracy_per_byte` sweep instead of a single headline accuracy.

Telemetry fields:

- `accuracy_per_byte`
- `accuracy_per_token`
- `budget_utilization`
- `protected_budget_fraction`
- `calibration_loss`
- `selection_confidence`
- `repair_confidence`
- `route_cost_curve`

Paper-safe note:

- `Dev-smoke` can use the current GSM30 pool and toy bridges to validate budget accounting.
- `Held-out` should report a Pareto curve, not a single point, and every compared method must use the same budget envelope.

## 5) Interpretable telemetry contract

If a LatentWire ablation is meant to be useful later, it should log at least:

- `route_id`
- `candidate_rank`
- `pre_answer`
- `post_answer`
- `oracle_answer`
- `selection_confidence`
- `repair_confidence`
- `route_entropy`
- `layer_bit_allocation`
- `retained_kv_fraction`
- `bytes_moved`
- `prompt_chars`
- `repair_token_budget`
- `help_rate`
- `harm_rate`

The goal is to make every improvement explainable as one of:

- better route selection
- better budget allocation
- better outlier handling
- better repair quality
- better reuse of compressed context

## Highest-priority LatentWire moves

1. Promote process repair from GSM30 dev-smoke into held-out GSM70 / SVAMP with a strict final-answer contract and matched budget.
2. Add mixed-bit and mixed-budget bridge ablations, because quantization literature gives the cleanest language for sensitivity-aware budget allocation.
3. Add query-aware versus query-agnostic KV compression comparisons, including head-level and layer-level budgets.
4. Report accuracy-per-byte and accuracy-per-token curves, not just raw accuracy.
5. Keep telemetry interpretable enough that each gain can be attributed to selection, smoothing, compression, or repair rather than to hidden budget inflation.
