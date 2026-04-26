# Qwen2.5-Math -> Qwen3 SVAMP70 Surface Scout Chal171-240

- date: `2026-04-26`
- status: source-sidecar surface rejected before C2C spend
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval slice: SVAMP `chal-171` through `chal-240`

## Start Status

- current ICLR readiness: not ready
- current story: source-sidecar routing is unstable across disjoint SVAMP
  surfaces
- exact blocker: find a surface with enough clean source-only IDs to support a
  meaningful sidecar/router gate

## Results

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| source-alone | 8/70 | 64/70 |
| target-alone | 22/70 | 70/70 |
| text relay | 24/70 | 70/70 |

Target/source oracle is `24/70`; source-only over target is only `2`, and
clean source-only after excluding text relay is only `1`.

## Decision

Reject this slice as a source-sidecar decision surface. It has too few clean
source-only IDs to test source-derived communication, and target/text are both
much stronger than source-alone. Do not spend C2C compute here for the current
sidecar branch.

This strengthens the diagnosis from the CV-router holdout: the next useful
work is source-surface discovery or a stronger source encoder, not more router
tuning on weak source-complementary slices.

## Next Gate

Search for a Qwen2.5-Math -> Qwen3 slice or benchmark subset where:

- source-only over target is at least `6/70`
- clean source-only after text exclusion is at least `4/70`
- target/source oracle is meaningfully above target
- then run C2C only if the slice has enough source-complementary mass

## Artifacts

- eval slice:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/_artifacts/svamp_chal171_240.jsonl`
  - sha256: `36fa351b699537cfdfd328035d95e7e50ca5fb23fb562bec8db90b1ff94127f5`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/manifest.md`
  - sha256: `145f425cd135d08272efe9cd7d0d973fb6b7e52ab744d960a390251d72ea1fc7`
- source-contrastive target set:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json`
  - sha256: `efe234a2e31ea60f4fa729b9493d48e7db736b2a95de25407f4902b1703f889e`

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/_artifacts/svamp_chal171_240.jsonl \
  --results-dir results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```
