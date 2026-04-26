# Qwen2.5-Math SVAMP32 Target-Query Source Bottleneck - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable source-necessary method plus medium,
  seed-repeat, strict source-control, and cross-family gates
- current story: C2C exposes clean target-missed headroom, but source-derived
  bottlenecks still do not recover the clean residual IDs
- blocker: even target-query-conditioned source attention does not produce
  clean C2C-residual recovery on the strict SVAMP32 gate

## Gate

This cycle implemented the next live branch after the Perceiver memory gate
failed: a strict pre-generation target-query-conditioned source bottleneck.

The model cross-fits a tiny classifier over frozen source and target token
states:

- target prompt states form query summaries
- target-conditioned queries attend over source token states
- the classifier predicts C2C residue signatures over moduli `2,3,5,7`
- candidate-pool decoding falls back to target-alone
- controls include zero-source, shuffled-source, label-shuffle,
  same-norm-noise, target-only-prefix, projected-soft-prompt, target-only, and
  slots-only

Promotion required:

- matched `>=10/32`
- target floor preserved versus target-only `8/32`
- clean source-necessary `>=2/6`
- control clean union `0/6`

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_target_query_source_bottleneck_probe.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --candidate source=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --fallback-label target \
  --moduli 2,3,5,7 \
  --query-count 4 \
  --hidden-dim 16 \
  --epochs 80 \
  --outer-folds 8 \
  --seed 1 \
  --shuffle-offset 1 \
  --min-correct 10 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 26 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --feature-layers mid,last \
  --device mps \
  --train-device mps \
  --dtype float32 \
  --date 2026-04-26 \
  --output-json results/qwen25math_svamp32_target_query_source_bottleneck_20260426/probe.json \
  --output-md results/qwen25math_svamp32_target_query_source_bottleneck_20260426/probe.md
```

## Evidence

| Condition | Correct | Clean Correct |
|---|---:|---:|
| matched | 7/32 | 0/6 |
| zero_source | 8/32 | 0/6 |
| shuffled_source | 6/32 | 0/6 |
| label_shuffled | 6/32 | 0/6 |
| same_norm_noise | 7/32 | 0/6 |
| target_only_prefix | 8/32 | 0/6 |
| projected_soft_prompt | 8/32 | 0/6 |
| target_only | 8/32 | 0/6 |
| slots_only | 6/32 | 0/6 |

Criteria:

- `min_correct`: failed
- `preserve_fallback_floor`: failed
- `min_clean_source_necessary`: failed
- `control_clean_union_empty`: passed

## Artifacts

- analyzer:
  - `scripts/analyze_svamp32_target_query_source_bottleneck_probe.py`
  - sha256: `7fcaa9901ea5a78e23e7d4af8a64f513c0cf9da40ab1697fc0ea88c719143203`
- tests:
  - `tests/test_analyze_svamp32_target_query_source_bottleneck_probe.py`
  - sha256: `3149ad0dbdedf7d03edf9004a0220626ede2d4c1dcd4ab075c10147dd8e2930a`
- result JSON:
  - `results/qwen25math_svamp32_target_query_source_bottleneck_20260426/probe.json`
  - sha256: `06141d71be5fc57230aa7346525731618f554b023d7230c794ab681c34b05280`
- readout:
  - `results/qwen25math_svamp32_target_query_source_bottleneck_20260426/probe.md`
  - sha256: `482a661d22065e93a83a0d9b2fb5cd5fb5c343d4d051a4eba70fc305bd7be9aa`
- log:
  - `.debug/qwen25math_svamp32_target_query_source_bottleneck_20260426.log`
  - sha256: `2c338d2c18ede9c25e2c4fc106bfa8d5a09524bb99fc2d5b5a495c6e1bb89636`

## Decision

Fail and kill this target-query-conditioned residue-classifier branch on the
current SVAMP32 C2C-headroom surface. It avoids clean control leakage, but it
does not recover any clean residual ID and falls below the target-only floor.

Do not tune query count, hidden dim, epochs, moduli, or layer selection on this
exact classifier branch without a new signal source.

## Next Gate

Stop residue-classifier/readout variants on this surface. The next learned
branch, if pursued, should be a true source-conditioned soft-prefix or gated
cross-attention objective trained directly on gold-vs-distractor logprob, with
matched target-only learned-prefix, slots-only learned-prefix, projected
soft-prompt, zero-source, and shuffled-source controls before generation.
