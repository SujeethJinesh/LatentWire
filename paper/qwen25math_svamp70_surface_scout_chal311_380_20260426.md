# Qwen2.5-Math -> Qwen3 SVAMP70 Surface Scout Chal311-380

- date: `2026-04-26`
- status: source-sidecar surface rejected before C2C spend
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval slice: SVAMP `chal-311` through `chal-380`
- scale-up rung: source-surface discovery

## Start Status

- current ICLR readiness: not ready
- current story: source-sidecar and confidence-router signals do not generalize
  from the original SVAMP70 live surface to disjoint holdout slices
- exact blocker: find enough clean source-only mass to justify C2C or learned
  connector compute
- highest-priority gate: source-only over target at least `6/70` and clean
  source-only after text exclusion at least `4/70`

## Results

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| source-alone | `8/70` | `63/70` |
| target-alone | `21/70` | `70/70` |
| text relay | `19/70` | `70/70` |

Target/source oracle is `24/70`; source-only over target is only `3`, and
clean source-only after excluding text relay is only `2`.

## Decision

Reject `chal311-380` as a source-sidecar decision surface. It fails both the
raw source-only threshold and the clean source-only threshold, so do not spend
C2C or connector compute here.

This is now the third adjacent SVAMP range scout with insufficient clean source
mass:

- `chal171-240`: source-only `2`, clean source-only `1`
- `chal241-310`: source-only `4`, clean source-only `4`, but raw source too low
- `chal311-380`: source-only `3`, clean source-only `2`

Stop adjacent SVAMP range scouting for this source/target pair unless a new
source encoder or prompting hypothesis changes the surface. The next branch
should either use a different source/target pair with a cheap source/target/text
scout first, or return to a stronger learned communication objective on the
best existing live/holdout surfaces.

## Artifacts

- eval slice:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.jsonl`
  - sha256: `f503455a810222bbc5652a58824c5f5090d6a9d7d80973eab2caac5d51612227`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/manifest.md`
  - sha256: `8b5d25ed0a2e4c83cd2889f9221185f6c7e5f5e201391871f5b1fbfe658eb516`
- source-contrastive target set:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json`
  - sha256: `f1bdc7e775075a2b40b7ed0c96cf039795185868a1722afba9047b70c6bd67dc`
- consolidated surface scan:
  - `results/source_headroom_surface_scan_20260426/scan_with_chal311.json`
  - sha256: `5f5034f7de04ffbf48b1dcc7dcac737fea736f17b8e9b0d7f6e8e70246bd10b4`

## Commands

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_jsonl_range.py \
  --source data/svamp_1000.jsonl \
  --output results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.jsonl \
  --start-index 311 --count 70 \
  --manifest-json results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.manifest.json \
  --manifest-md results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.manifest.md
```

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.jsonl \
  --results-dir results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 70 --device mps --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis --use-chat-template --no-enable-thinking \
  --continue-on-error
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/build_source_contrastive_target_set.py \
  --target target=path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl,method=source_alone \
  --baseline t2t=path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/text_to_text.jsonl,method=text_to_text \
  --min-source-only 6 \
  --output-json results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json \
  --output-md results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.md
```

## Next Exact Gate

Do not continue adjacent SVAMP range scouting. Pick one of:

- a different source/target pair with a cheap source/target/text scout first
- a stronger source encoder/prompting hypothesis, then rerun a source-surface
  scout
- a bounded learned communication objective on the existing best live/holdout
  surfaces, with full source-destroying controls before generation
