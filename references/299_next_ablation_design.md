# Next Dynalign Ablation Design

Goal: identify the smallest code change after `bridge_ridge_qk_dynalign_spanalm_module_replace` that is still materially different from the current span-ALM teacher, while staying inside the existing dynalign / query-conditioned module path.

The short version: **do not add another bridge family**. Reuse the current dynalign span-ALM path, but make the teacher stronger in one of two ways:

1. **attention/refinement-distilled dynalign**: turn on attention-side refinement loss in the existing query-conditioned residual fitter; or
2. **query-routed dynalign**: keep the same span-ALM teacher, but route loss weight through the existing sample-weight path with a more explicit query-confidence signal.

The smallest and most defensible first step is **attention/refinement-distilled dynalign**.

## Why this lane

Current span-ALM already does the following:

- dynamic remapping via `collect_dynamic_prompt_position_mixtures(...)`;
- approximate teacher supervision via `collect_aligned_prediction_teacher(...)`;
- sample weighting via `collect_alignment_confidence_weights(...)` and `collect_prediction_confidence_weights(...)`;
- direct-output module replacement via `bridge_ridge_qk_dynalign_spanalm_module_replace`.

So the next incremental question is not whether the teacher can be made token-aware. It is whether the bridge benefits from a **stronger refinement signal** that conditions the fit on query/attention structure, not only on span likelihood.

## Recent primary sources

| Paper | Date | Link | What to steal |
|---|---:|---|---|
| Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking | 2025-06-11 | https://arxiv.org/abs/2506.09944 | Query-conditioned attention mass is a better runtime signal than a fixed offline head prior; use query-aware routing instead of a single static teacher weight. |
| Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution | 2025-10-01 | https://arxiv.org/abs/2510.00636 | Runtime importance should be estimated from future-query statistics, not from one static alignment fit. |
| AttAnchor: Guiding Cross-Modal Token Alignment in VLMs with Attention Anchors | 2025-09-27 | https://arxiv.org/abs/2509.23109 | Attention anchors can make alignment losses reflect content locality instead of only token identity. |
| Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions | 2026-02-10 | https://arxiv.org/abs/2602.09483 | Token interaction distillation is a stronger teacher signal than plain next-token matching; use it as the next refinement objective. |

## Recommended branch

### `bridge_ridge_qk_dynalign_attnrefine_module_replace`

This should be the next ablation target if we want the smallest materially different step after span-ALM.

The implementation should reuse the current dynalign span-ALM plumbing, but change the fit loss in the module fitter:

- keep the same dynamic span mixture construction;
- keep the same approximate-likelihood teacher;
- keep the same sample-weight mixture from alignment confidence + prediction confidence;
- add an attention/refinement term in the fitter itself.

Concretely, the target is the existing `_fit_bridge_query_residual_adapter(...)` path in `latent_bridge/translator.py`, not a new module architecture.

### Why this is materially different from span-ALM

Span-ALM already asks: "can a small span-window teacher rescue the bridge?"

This next step asks a different question:

> does the bridge improve when the teacher also forces the student to match the **query-conditioned attention geometry** or a related refinement signal, rather than only span likelihood mass?

That is a stronger supervision target than the current span-only approximate-likelihood setup.

## Exact hook points

### `latent_bridge/calibrate.py`

Touch the existing dynalign branch logic in `main()` only:

- the branch that already handles
  - `bridge_ridge_qk_dynalign_ctxonly_module_replace`
  - `bridge_ridge_qk_dynalign_module_replace`
  - `bridge_ridge_qk_dynalign_dwakd_module_replace`
  - `bridge_ridge_qk_dynalign_likelihood_module_replace`
  - `bridge_ridge_qk_dynalign_spanalm_module_replace`
  - `bridge_ridge_qk_dynalign_interact_module_replace`
- the span-ALM teacher builder near the current call to `collect_aligned_prediction_teacher(...)`
- the current sample-weight builder:
  - `collect_alignment_confidence_weights(...)`
  - `collect_prediction_confidence_weights(...)`
  - `translator.set_bridge_sample_weights(...)`

Minimal change:

1. keep the current span-ALM teacher;
2. add one new alias / branch condition for `bridge_ridge_qk_dynalign_attnrefine_module_replace`;
3. keep the same combined confidence weights;
4. pass the stronger refinement signal through to the translator branch.

### `latent_bridge/translator.py`

The smallest usable fit change is already available in:

- `_fit_bridge_query_residual_adapter(...)`

This fitter already supports the relevant supervision knobs:

- `attention_kl_weight`
- `prediction_distill_weight`
- `dynamic_prediction_weight`
- `sample_weights`
- `interaction_distill_weight`
- `readout_distill_weight`

So the smallest code change is to have the dynalign span-ALM branch call this fitter with:

- the existing `prediction_distill_weight` / `dynamic_prediction_weight`
- plus a nonzero `attention_kl_weight`

If we want the query-routed version instead, the same fitter can reuse `sample_weights` more aggressively without changing the model family.

### `latent_bridge/evaluate.py`

No evaluator math change is needed.

Only touch `evaluate.py` if we want:

- a new method label in result summaries; or
- a new CLI alias for the branch name.

Keep the current fairness controls untouched:

- `--source-use-chat-template`
- `--target-use-chat-template`
- `--source-enable-thinking false`
- `--target-enable-thinking false`

## What to reuse

- `collect_dynamic_prompt_position_mixtures(...)`
- `collect_aligned_prediction_teacher(...)`
- `collect_alignment_confidence_weights(...)`
- `collect_prediction_confidence_weights(...)`
- `translator.set_bridge_sample_weights(...)`
- the current `bridge_ridge_qk_dynalign_spanalm_module_replace` route
- `_fit_bridge_query_residual_adapter(...)`

Do **not** introduce a new bridge family or a new transport path.

## What to avoid changing

- grouped transport math
- tokenizer / vocab surgery
- the transport-bit accounting
- `eval_rotalign_kv(...)`
- Qwen thinking-mode controls, except for making them consistent across source and target

## Expected telemetry

Log the following for calibration and held-out evaluation:

### Calibration telemetry

- aligned samples per prompt
- mean / max span coverage
- teacher entropy
- teacher top-k mass
- sample-weight mean / std / min / max
- attention-KL loss value
- dynamic teacher KL value
- prompt-level truncation rate

### Held-out telemetry

- exact-match accuracy
- delta vs span-ALM baseline
- delta vs target-alone
- McNemar p-value
- bootstrap delta CI
- average bytes / example
- latency per example

## Expected failure modes

1. **Attention refinement collapses to the same signal as span-ALM.** The new term may not change the fit enough to move held-out accuracy.
2. **Sample weights become a prompt-length proxy.** If the query-routing signal just tracks length or token count, the branch will look better offline but stay flat on held-out.
3. **Attention KL over-regularizes calibration.** The bridge may become too faithful to the teacher geometry and lose the useful residual flexibility.
4. **The teacher remains too token-bound.** If the span-level teacher is still too exact-token dependent, the issue is the interface, not the loss.
5. **Serialization confounds remain.** If prompt formatting or thinking-mode differ across source and target, the branch can appear to fail for the wrong reason.

## Suggested run ladder

1. Current baseline: `bridge_ridge_qk_dynalign_spanalm_module_replace`
2. Next ablation: `bridge_ridge_qk_dynalign_attnrefine_module_replace`
3. Query-routed variant: same branch, but with stronger use of the existing confidence-weight path

## Commands to run

Use the same fair Qwen control already used elsewhere in the repo:

```bash
python scripts/calibrate.py \
  --calibration-file data/gsm8k_gate_search_30.jsonl \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --quantization-correction bridge_ridge_qk_dynalign_spanalm_module_replace \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device cuda \
  --dtype bfloat16
```

After the new alias is wired, rerun the same command with:

```bash
--quantization-correction bridge_ridge_qk_dynalign_attnrefine_module_replace
```

Then evaluate on the held-out set with the same prompt controls.

## Bottom line

The smallest meaningful increment after span-ALM is **attention/refinement-distilled dynalign** on top of the existing `bridge_ridge_qk_dynalign_spanalm_module_replace` path. That keeps the teacher tokenizer-agnostic, preserves the current confidence-weight plumbing, and tests whether the bridge needs a stronger query-conditioned refinement signal rather than another span-only loss.
