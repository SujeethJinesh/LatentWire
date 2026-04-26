# Qwen2.5-Math -> Qwen3 SVAMP70 Source Confidence Router

- date: `2026-04-26`
- status: `fails_live_holdout_gate`
- scale-up rung: medium live-CV plus frozen holdout falsification
- live branch entering run: source-internal confidence router over source-only
  greedy-generation diagnostics

## Question

Decoded source-text guards and trace routers failed. This run asks whether
source-internal confidence signals from the source model's greedy generation
provide a more reliable router feature family.

The diagnostics collect chosen-token logprob, entropy, top-1 probability, and
top-1/top-2 logit margin without mutating existing baseline prediction files.
The router is intentionally small: a one-feature decision stump trained by
5-fold live CV, then a full-live frozen rule applied once to the disjoint
SVAMP70 holdout.

## Diagnostics

Live diagnostics:

- eval file:
  `results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl`
- source correct: `13/70`, matching the existing `source_alone` baseline
- JSONL sha256:
  `b17755be3db764f6130830cc516b18b6e4fadce7a78de36d20f10dd8c84c69b2`

Holdout diagnostics:

- eval file:
  `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl`
- source correct: `8/70`, matching the existing holdout `source_alone` baseline
- JSONL sha256:
  `2fc5226940ea4fc743324534bb51c938829910810619040f78afea2c905ecb0e`

## Router Command

```bash
./venv_arm64/bin/python scripts/analyze_source_confidence_router_gate.py \
  --live-diagnostics-jsonl results/qwen25math_svamp70_source_generation_diagnostics_20260426/source_diagnostics.jsonl \
  --live-target-jsonl results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl \
  --live-target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-diagnostics-jsonl results/qwen25math_svamp70_source_generation_diagnostics_20260426/holdout_source_diagnostics.jsonl \
  --holdout-target-jsonl results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl \
  --holdout-target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --target-method target_alone \
  --outer-folds 5 \
  --accept-penalty 0.10 \
  --output-json results/qwen25math_svamp70_source_generation_diagnostics_20260426/confidence_router.json \
  --output-md results/qwen25math_svamp70_source_generation_diagnostics_20260426/confidence_router.md
```

## Result

Frozen full-live rule:

- feature: `min_top1_prob`
- direction: `ge`
- threshold: `0.1639411821961403`
- train help: `4`
- train harm: `0`
- train accept: `8`

Live CV:

- matched correct: `24/70`
- clean source-necessary: `2`
- clean control union: `0`
- accepted harm: `0`
- failing criteria: `min_correct`, `min_clean_source_necessary`

Holdout frozen:

- matched correct: `7/70`
- clean source-necessary: `0`
- clean control union: `0`
- accepted harm: `1`
- failing criteria: `min_correct`, `min_clean_source_necessary`

## Decision

Prune source-internal confidence routing as a positive method on this surface.
The live rule is cleaner than shallow decoded-text guards, but it is too weak
under live CV and does not generalize to holdout. Do not tune multi-feature
confidence routers on this surface unless a new source surface has more clean
source-only mass.

The next highest-value branch is the disjoint `chal311-380` source-surface
scout with source/target/text only. Spend C2C or connector compute only if that
slice reaches at least `6/70` source-only over target and at least `4/70` clean
source-only after text exclusion.

## Artifacts

- manifest:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/manifest.md`
- confidence router:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/confidence_router.json`
  - sha256: `291ee7015a7b28f41f7c5e1b397e18b29da1b0781ae0f30c7c528ac3e860b4a8`
- confidence router readout:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/confidence_router.md`
  - sha256: `6f574b2864036520da9d769c3c4aac875f13ba3a9df65a7ca21dc5b4994baaa7`
- diagnostics:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/source_diagnostics.jsonl`
  - sha256: `b17755be3db764f6130830cc516b18b6e4fadce7a78de36d20f10dd8c84c69b2`
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/holdout_source_diagnostics.jsonl`
  - sha256: `2fc5226940ea4fc743324534bb51c938829910810619040f78afea2c905ecb0e`

## Tests

- `./venv_arm64/bin/python -m pytest tests/test_collect_source_generation_diagnostics.py tests/test_analyze_source_confidence_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/collect_source_generation_diagnostics.py scripts/analyze_source_confidence_router_gate.py`
