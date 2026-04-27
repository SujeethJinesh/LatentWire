# SVAMP32 Source Sampling Reachability

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready; still missing a deployable, source-derived
   method that survives strict controls and seed/rung confirmation.
2. Current paper story: target/no-source candidate generation gives receiver
   headroom, and source-side generation exposes different residual answers, but
   neither is yet communication.
3. Exact blocker to submission: matched source must cause a target receiver to
   select or generate useful information that source-destroying controls cannot
   reproduce.
4. Current live branches: source-conditioned candidate generation plus strict
   selector; JEPA-style answer-masked connector with anti-collapse telemetry.
5. Highest-priority gate this cycle: test whether source sampling exposes C2C
   clean residual candidates beyond the full32 target/no-source pool.
6. Scale-up rung: smoke.

## Harness Update

- `scripts/sample_target_candidate_surface.py`
  - added `--prompt-mode {direct,source_reasoning}`
  - added `--source-reasoning-mode`
  - added `--method-prefix` for future source-vs-target row names
  - records prompt-mode metadata and prompt token telemetry for the actual
    sampled prompt
- `scripts/compare_candidate_pool_reachability.py`
  - compares two reachability audits
  - reports total oracle deltas, new/lost oracle IDs, and new C2C-clean
    residual IDs

## Commands

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --model Qwen/Qwen2.5-Math-1.5B \
  --samples 4 \
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
  --output-jsonl results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl \
  --output-json results/svamp32_source_sampling_full32_s4_20260427/source_samples.json \
  --output-md results/svamp32_source_sampling_full32_s4_20260427/source_samples.md

./venv_arm64/bin/python scripts/analyze_target_sampling_reachability.py \
  --samples-jsonl results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl \
  --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json \
  --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --date 2026-04-27 \
  --output-json results/svamp32_source_sampling_full32_s4_20260427/reachability.json \
  --output-md results/svamp32_source_sampling_full32_s4_20260427/reachability.md

./venv_arm64/bin/python scripts/compare_candidate_pool_reachability.py \
  --baseline-reachability results/svamp32_target_sampling_full32_s8_20260427/reachability.json \
  --candidate-reachability results/svamp32_source_sampling_full32_s4_20260427/reachability.json \
  --date 2026-04-27 \
  --output-json results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.json \
  --output-md results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.md
```

## Results

- source-sampled candidate oracle: `10/32`
- target/no-source full32 S8 baseline oracle: `14/32`
- source minus target/no-source oracle: `-4`
- new source oracle IDs beyond target/no-source: `5`
- lost target/no-source oracle IDs: `9`
- C2C clean residual in source pool: `3/6`
- new C2C-clean residual IDs beyond target/no-source: `2`
  - `6e9745b37ab6fc45`
  - `de1bf4d142544e5b`
- C2C teacher-only IDs in source pool: `4/9`
- mean unique sampled answers per ID: `3.406`
- duplicate nonempty row fraction: `0.148`

## Decision

This is a source-surface discovery pass and a method-evidence fail. The source
model is worse than target/no-source sampling in total oracle reachability, so
there is no accuracy claim. However, the source-sampled pool reaches two new
C2C-clean residual gold answers that target/no-source full32 S8 did not reach.

The next gate is therefore not another broad target-only sampling run. It is a
strict matched-source selector or connector gate on the two new IDs. Promotion
requires at least one matched-only clean recovery on `6e9745b37ab6fc45` or
`de1bf4d142544e5b`, control clean union `0`, no target-correct harm, and bytes
plus collapse telemetry.

## JEPA / Anti-Collapse Consequence

JEPA, V-JEPA, LeJEPA/SIGReg, VICReg, and Barlow Twins remain design constraints:
use answer-masked source views, predict only source innovation over a frozen
target-prior baseline, and report variance/effective-rank/covariance telemetry.
They do not become a method claim until matched source beats zero, shuffled,
target-only/slots-only, answer-only, answer-masked, and random same-byte
controls on the new source-only residual IDs.

## Artifacts

- `results/svamp32_source_sampling_full32_s4_20260427/manifest.md`
- `results/svamp32_source_sampling_full32_s4_20260427/source_samples.md`
- `results/svamp32_source_sampling_full32_s4_20260427/reachability.md`
- `results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.md`
