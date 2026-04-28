# Prompt-Wrapper Source Surface Controls

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready; no source-derived method has survived
   target-prior and source-destroying controls.
2. Current paper story: target prompt wrappers are a strong no-source candidate
   generator, and the prior SVAMP32 source-sampling clue was prompt-prior
   explained.
3. Exact blocker to submission: find source-derived residual IDs beyond target
   direct, target brief-wrapper, no-source merged pools, answer leakage, and
   source-destroying controls.
4. Current live branches: prompt-controlled source-surface discovery; JEPA-style
   source-innovation connectors only after a target-prior-unexplained surface
   exists.
5. Highest-priority gate: test reusable SVAMP70/GSM clean source-only surfaces
   against target prompt-wrapper controls.
6. Scale-up rung: strict small gate.

## Subagent Integration

- planner: skip SVAMP32 frontier because target direct plus target brief S4
  already covers `6/6` C2C-clean residual IDs; try GSM clean2 if a source
  surface remains after prompt controls.
- reviewer: define source residual IDs only after subtracting target wrappers
  up to matched or larger budget, prompt-format controls, no-source merged
  pools, selector controls, and source answer leakage.
- JEPA/anti-collapse: keep Query-JEPA, masked target-state fill-in, and
  dual-view source-innovation consistency as deferred designs only.
- artifact audit: best reusable larger surface is Math-7B SVAMP70 clean7; run
  target brief-wrapper S8 before any connector work.

## Gate 1: Math-7B SVAMP70 Clean7

Command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/_artifacts/svamp_eval_70_70.jsonl \
  --target-set-json results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json \
  --id-fields clean_source_only \
  --output-jsonl results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.jsonl \
  --output-meta-json results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.meta.json

HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 8 \
  --method-prefix target_brief_sample \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 71 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --prompt-mode source_reasoning \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.jsonl \
  --output-json results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.json \
  --output-md results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.md
```

Result:

- target brief-wrapper S8 oracle: `4/7`
- target-prior explained IDs: `3c5aeb08941dbb6d`,
  `ce08a3a269bf0151`, `de1bf4d142544e5b`, `e099e405e8d1a66b`
- residual candidate IDs after target brief S8: `33836927fc9f1a8a`,
  `4c84ebf42812703b`, `d64f6e35083ffe8c`

Decision: partial prune. This surface is not dead, but most of the clean
source-only label was actually target prompt-wrapper reachability. The remaining
three IDs need answer-masked source controls before a connector can be trained.

## Gate 2: GSM Clean2

Commands:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file results/qwen25math_qwen3_gsm70_source_surface_20260426/_artifacts/gsm8k_eval_70_70.jsonl \
  --target-set-json results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json \
  --id-fields clean_source_only \
  --output-jsonl results/gsm_source_residual_prompt_control_clean2_20260427/gsm_clean2_eval.jsonl \
  --output-meta-json results/gsm_source_residual_prompt_control_clean2_20260427/gsm_clean2_eval.meta.json
```

Then source brief S8, target direct S16, and target brief S16 were sampled with
`scripts/sample_target_candidate_surface.py` and audited with
`scripts/analyze_target_sampling_reachability.py`.

Result:

- source brief S8 oracle: `1/2`
- target direct S16 oracle: `1/2`
- target brief-wrapper S16 oracle: `1/2`
- target direct plus target brief-wrapper union oracle: `1/2`
- source addition beyond target prompt union: `0`

Decision: fail. The only source-reached ID, `1deed634dcd7d229`, is also reached
by target direct and target brief-wrapper sampling. GSM clean2 is not a live
source-communication surface.

## JEPA / Anti-Collapse Consequence

JEPA, LeJEPA, V-JEPA, VICReg, and Barlow Twins remain useful for future
connector design, but only after a target-prior-unexplained residual surface
exists. The immediate diagnostic to add to future discovery runs is a
target-prior-subtracted residual table with finite coverage, effective rank,
variance floor, covariance off-diagonal mass, slot/query entropy, and
matched-vs-zero/shuffled/answer-only margins.

## Next Exact Gate

Run answer-masked and answer-only source controls on the three remaining
Math-7B SVAMP70 residual candidates:

- `33836927fc9f1a8a`
- `4c84ebf42812703b`
- `d64f6e35083ffe8c`

Pass requires at least `2/3` matched-source recoveries beyond target direct plus
target brief-wrapper, `0` clean recovery from answer-only/answer-masked leakage
controls, and `0` target-correct harms before any JEPA/query connector work.
