# Span-ALM Teacher Plan for LatentWire

Goal: test whether a tokenizer-agnostic, span-level teacher can rescue the
current cross-tokenizer branch after the gold next-token likelihood boost
failed.

This memo stays implementation-oriented and reuses the current
`spanalign` / `ctxalign` / `dynalign` / `module_replace` plumbing instead of
introducing a new bridge family.

## Why this lane

The current readout says:

- raw prompt-span remapping improved offline fit, but held-out accuracy stayed
  flat;
- contextual token mixtures and output-aware dynalign are the first live
  remapping lane;
- direct next-token likelihood injection over-anchors and regresses to the
  context-only null.

So the next useful ablation is **not** another gold next-token boost. The next
minimal step is a span-level teacher that keeps the tokenizer boundary
problematized but removes dependence on exact token-ID correspondence.

## Recent primary sources

| Paper | Date | Link | What to steal |
|---|---:|---|---|
| Cross-Tokenizer Distillation via Approximate Likelihood Matching | 2025-03-25 | https://arxiv.org/abs/2503.20083 | Pure cross-tokenizer distillation without next-token prediction as the main objective; closest reference for approximate-likelihood supervision. |
| CoT2Align: Cross-Chain of Thought Distillation via Optimal Transport Alignment for Language Models with Different Tokenizers | 2025-02-24 | https://arxiv.org/abs/2502.16806 | OT alignment over sequence/layer structure for different tokenizers; useful if the span teacher needs a sequence-level alignment objective. |
| Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models | 2024-12-19 | https://arxiv.org/abs/2412.14528 | Token-level plus sequence-level OT with Sinkhorn distance; good fallback if span teacher needs a globally normalized alignment loss. |
| TokAlign: Efficient Vocabulary Adaptation via Token Alignment | 2025-06-04 | https://arxiv.org/abs/2506.03523 | One-to-one token-ID remap plus progressive fine-tuning; relevant if we later want a tokenizer bridge, but heavier than the span teacher. |
| CTPD: Cross Tokenizer Preference Distillation | 2026-01-17 | https://arxiv.org/abs/2601.11865 | Aligned span projection on shared character-level spans; strongest recent reference for a span-level teacher with heterogeneous tokenizers. |
| AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation | 2025-03-25 | https://arxiv.org/abs/2503.19693 | Lightweight vocabulary adaptation; useful only as a later tokenizer-control baseline, not the next LatentWire bridge step. |

## Minimal LatentWire ablation

### Recommended branch

Run a **span-projected direct-output module replace**:

- `bridge_ridge_qk_spanalign_module_replace` as the base branch;
- use a span-aware alignment helper instead of exact token-position pairing;
- keep the existing teacher machinery, but feed it aligned span projections
  rather than gold next-token likelihood as the main signal.

If we want the strictest tokenizer-agnostic version, this is the branch that
should become `bridge_ridge_qk_bytespan_module_replace` later. For now, the
cheap path is to keep the current `spanalign` entrypoint and change only the
pairing helper.

### Why this is materially different from the failed gold-next-token boost

- Gold next-token KL assumes exact token boundary compatibility.
- Span-ALM style supervision only assumes that teacher/student outputs can be
  projected onto shared spans or aligned span mixtures.
- That means the bridge can still learn from teacher likelihood structure even
  if the tokenizers disagree on chunking.

### Smallest ablation ladder

1. `bridge_ridge_qk_spanalign_module_replace` with raw span alignment.
2. `bridge_ridge_qk_ctxalign_module_replace` with context-weighted span
   mixtures.
3. `bridge_ridge_qk_dynalign_module_replace` as the stronger output-aware
   upper bound.
4. `bridge_ridge_qk_dynalign_likelihood_module_replace` only as the explicit
   approximate-likelihood check, because the current notes already suggest that
   naive target-next-token injection over-anchors.

## Exact hook points

### `latent_bridge/calibrate.py`

Touch these functions only:

- `collect_aligned_prompt_position_pairs(...)`
- `collect_contextual_prompt_position_mixtures(...)`
- `collect_dynamic_prompt_position_mixtures(...)`
- `collect_dynamic_program_prompt_position_pairs(...)`
- `collect_aligned_prediction_teacher(...)`
- `collect_aligned_prompt_valid_lengths(...)`
- `collect_aligned_query_features(...)`

The cheapest implementation path is:

1. add a byte/span-aware helper next to `collect_aligned_prompt_position_pairs(...)`;
2. reuse `collect_aligned_prediction_teacher(...)` unchanged;
3. route the new helper into the existing `spanalign` / `ctxalign` / `dynalign`
   calibration branches.

### `latent_bridge/translator.py`

Reuse these fitters as-is:

- `_fit_bridge_query_module_replace(...)`
- `_fit_bridge_query_tokenbasis_replace(...)`

Only add a new alias / dispatch branch if you want a new name for the
byte-span path. Do not introduce a new bridge family.

### `latent_bridge/evaluate.py`

Avoid changing the evaluator math.

Only reuse the existing prompt serialization controls:

- `_format_prompt_for_tokenizer(...)`
- `_prepare_prefix_state(...)`
- `_build_rotalign_prefix_state(...)`
- `eval_rotalign_kv(...)`
- `_eval_generation_rotalign_with_stats(...)`

The fairness controls should stay:

- `--source-use-chat-template`
- `--target-use-chat-template`
- `--source-enable-thinking false`
- `--target-enable-thinking false`

## What to avoid changing

- Do not touch the grouped transport math.
- Do not touch the quantization channel or the transport bits accounting.
- Do not introduce tokenizer surgery or vocab replacement.
- Do not change `eval_rotalign_kv(...)` scoring.
- Do not change the Qwen thinking-mode controls beyond making them consistent
  across source and target.

## Expected failure modes

1. **Span coverage is too sparse.** If the raw-span alignment yields too few
   usable pairs, the teacher will look strong offline and still fail held-out.
2. **Span mixtures are too diffuse.** Context mixtures may improve fit but wash
   out the likelihood signal, collapsing back to the context-only null.
3. **Serialization confound dominates.** If `enable_thinking` or chat-template
   mismatch is still active, any teacher gain will be misattributed.
4. **Likelihood is still too token-bound.** If span projection cannot salvage
   the objective, the issue is probably not the teacher target but the bridge
   interface itself.

## Metrics to log

For the calibration run:

- aligned pairs / prompt;
- mean target spans per source span;
- truncation rate;
- fit `K` cosine / `V` cosine;
- teacher entropy / top-k mass;
- teacher span coverage;
- runtime bits / bytes.

For held-out evaluation:

- exact-match accuracy;
- paired delta vs target-alone and vs the current gold-next-token teacher;
- McNemar p-value;
- bootstrap delta CI;
- avg bytes / latency.

## Commands to run

Use the same Qwen pair already working in the repo:

```bash
python scripts/calibrate.py \
  --calibration-file data/gsm8k_gate_search_30.jsonl \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --quantization-correction bridge_ridge_qk_spanalign_module_replace \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device cuda \
  --dtype bfloat16
```

```bash
python scripts/evaluate.py \
  --eval-file data/gsm8k_eval_70.jsonl \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target,t2t,rotalign,rotalign_translated,rotalign_fused \
  --quantization-correction bridge_ridge_qk_spanalign_module_replace \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device cuda \
  --dtype bfloat16
```

If the span-aligned branch is live, the next follow-up is:

- `bridge_ridge_qk_ctxalign_module_replace` on the same calibration/eval pair;
- then `bridge_ridge_qk_dynalign_module_replace` as the stronger upper bound.

## Bottom line

If gold next-token likelihood boosting failed, the next credible step is a
**span-level approximate-likelihood teacher** with tokenizer-agnostic span
projection. That is the smallest ablation that still tests whether the failure
is really about token boundary mismatch rather than bridge capacity.
