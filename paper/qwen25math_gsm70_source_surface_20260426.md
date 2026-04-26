# Qwen2.5-Math -> Qwen3 GSM70 Source Surface Scout

- date: `2026-04-26`
- status: source-sidecar surface rejected before C2C spend
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval slice: `data/gsm8k_eval_70.jsonl`
- git commit before run: `3d6341424d11b2761beaebe6a029c311df85b877`

## Start Status

- current ICLR readiness: not ready
- current story: SVAMP source-sidecar evidence is unstable across disjoint
  slices, and repeated source-surface scouts have not found a stronger stable
  surface
- exact blocker: find enough clean source-only mass to justify C2C or learned
  connector compute

## Results

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| source-alone | 3/70 | 63/70 |
| target-alone | 4/70 | 70/70 |
| text relay | 6/70 | 70/70 |

Target/source oracle is `7/70`; source-only over target is `3`, and clean
source-only after excluding text relay is only `2`.

## Decision

Reject GSM70 as the next source-sidecar decision surface. It fails both the raw
source-only threshold and the clean source-only threshold. Do not run C2C or a
sidecar/router gate here for the current branch.

This weakens continued same-pair source-surface scouting as the main path.
The next branch should either:

- strengthen the source-derived interface on the best existing SVAMP surfaces,
  using a rate-capped query/resampler or shared sparse/anchor-relative code
- or change the source/target pair only with a clear hypothesis and cheap
  source/target/text scout first

## Next Gate

Implement or run the smallest real-model smoke for a stronger source interface
on an existing exact-ID surface:

- train or evaluate a bounded source-derived query/resampler or sparse sidecar
- include target-only, source-shuffled, zero-source, and text-relay controls
- require at least `2` clean source-necessary recoveries on SVAMP32 or at least
  `4` on SVAMP70 with control clean union `0`

## Artifacts

- materialized eval:
  - `results/qwen25math_qwen3_gsm70_source_surface_20260426/_artifacts/gsm8k_eval_70_70.jsonl`
  - sha256: `7d7a342b4c520553fdcf2e1159a5ed8f5627d238dce62a382761c4b6d8ab2dfc`
- generation manifest:
  - `results/qwen25math_qwen3_gsm70_source_surface_20260426/manifest.md`
  - sha256: `01045e3628480a2c2ba47f925e76c267c8e63f8efdf26c37635f740323c54fe6`
- source-contrastive target set:
  - `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json`
  - sha256: `ca8ed8e581c5aea1088071a0f69c19eb00c516cb8b7d2b0503b5f17aac65e061`

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/gsm8k_eval_70.jsonl \
  --results-dir results/qwen25math_qwen3_gsm70_source_surface_20260426 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```
