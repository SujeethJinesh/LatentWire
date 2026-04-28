# SVAMP32 Target Brief Wrapper Reachability

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready; no source-derived method has survived
   target-prior and source-destroying controls.
2. Current paper story: the last source-sampling clue was explained by target
   brief-wrapper prompting on the two-ID replay.
3. Exact blocker to submission: find source-derived residual IDs beyond target
   direct, target brief-wrapper, no-source merged pools, and source-destroying
   controls.
4. Current live branches: prompt-controlled source-surface discovery; compact
   source-innovation sidecars only after a surface survives prompt controls.
5. Highest-priority gate: full SVAMP32 target brief-wrapper S4 baseline compared
   against prior target direct S8 and source brief S4 surfaces.
6. Scale-up rung: smoke.

## Commands

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 4 \
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
  --output-jsonl results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.jsonl \
  --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.json \
  --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.md
```

The run was audited with `scripts/analyze_target_sampling_reachability.py`,
pairwise compared with `scripts/compare_candidate_pool_reachability.py`, and
unioned with the new `scripts/summarize_reachability_union.py`.

## Results

- target brief-wrapper S4 oracle: `18/32`
- target direct S8 oracle: `14/32`
- source brief S4 oracle: `10/32`
- target brief-wrapper C2C-clean residual IDs: `4/6`
  - `1d50b408c8f5cd2c`
  - `47464cc0b064f172`
  - `6e9745b37ab6fc45`
  - `de1bf4d142544e5b`
- target direct plus target brief-wrapper union:
  - oracle: `23/32`
  - C2C-clean residual reachability: `6/6`
- adding source brief S4 to the target-prior union:
  - oracle: `24/32`
  - new oracle ID: `b1200c32546a34a5`
  - new C2C-clean residual IDs: `0`

## Decision

The target brief-wrapper is a powerful target-prior candidate generator and must
be a mandatory baseline. The source brief S4 surface adds no C2C-clean residual
IDs beyond the union of target direct S8 and target brief S4. This prunes the
current source-sampling family as a communication surface.

## Next Exact Gate

Do not train a connector on the source-sampling surface. The next source-surface
discovery run must subtract target direct, target brief-wrapper at matched or
larger sample budget, no-source merged pools, answer-only/answer-masked source,
zero-source, shuffled-source, and random same-byte controls before any method
branch is promoted.

## Subagent Integration

- Planner: recommended full32 target brief-wrapper S4 first, S8 only if S4 was
  inconclusive. S4 was decisive.
- Reviewer: future source residual IDs must be defined after subtracting
  target brief-wrapper budgets up to the source budget.
- Creative scout: proposed causal-order, innovation matched-filter, and
  spread-spectrum challenge sidecars, but only after a surface survives
  prompt-wrapper controls.

## Artifacts

- `results/svamp32_target_brief_sampling_full32_s4_20260427/manifest.md`
- `results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.md`
- `results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.md`
- `results/svamp32_target_brief_sampling_full32_s4_20260427/source_addition_vs_target_prior_union.md`
