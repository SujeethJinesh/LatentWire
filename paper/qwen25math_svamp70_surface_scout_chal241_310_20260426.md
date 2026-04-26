# Qwen2.5-Math -> Qwen3 SVAMP70 Surface Scout Chal241-310

- date: `2026-04-26`
- status: weak source-complementary surface; do not spend C2C yet
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval slice: SVAMP `chal-241` through `chal-310`
- git commit before run: `7214b957eab0815f060f4daa2ec1bc22e30a55b9`

## Start Status

- current ICLR readiness: not ready
- current story: source-sidecar routing has one positive SVAMP70 surface, but
  shallow decoded-feature guards do not transfer reliably
- exact blocker: find a stable source-complementary surface before spending
  C2C, connector, or larger-run compute

## Results

| Method | Correct | Numeric Coverage |
|---|---:|---:|
| source-alone | 5/70 | 63/70 |
| target-alone | 10/70 | 70/70 |
| text relay | 14/70 | 70/70 |

Target/source oracle is `14/70`; source-only over target is `4`, and clean
source-only after excluding text relay is also `4`.

## Decision

Do not promote this slice to a C2C or sidecar decision run. It has a nonzero
clean source-only set, but it misses the predefined `>=6/70` source-only
headroom threshold and source-alone is far below text relay. The slice is useful
as a weak positive surface for diagnostics, not as the next medium gate.

The ranked surface scan now shows:

- original SVAMP70: strong raw source-complementary surface, `9` source-only
- holdout `chal-101..170`: strong raw source-complementary surface, `6`
  source-only, but text/control sidecar failed
- `chal-241..310`: weak raw surface, `4` source-only
- `chal-171..240`: weak raw surface, `2` source-only

## Hypothesis Update

- weakened: repeated SVAMP range scouting as the main path; source-only mass is
  sparse and unstable across adjacent slices
- strengthened: the next method should use a stronger source-derived interface
  or move to a different math surface before spending C2C
- live next branch: Qwen2.5-Math -> Qwen3 GSM70 source-surface scout, then
  rate-capped query/resampler or shared sparse sidecar only if the surface has
  enough clean source-only mass

## Next Gate

Run the GSM70 source-surface scout:

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

Run C2C only if source-only over target is at least `6/70` and clean
source-only after text exclusion is at least `4/70`.

## Artifacts

- range materializer:
  - `scripts/materialize_jsonl_range.py`
  - test: `tests/test_materialize_jsonl_range.py`
- eval slice:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310.jsonl`
  - sha256: `29b0ebf00df9c0bd552bc0dd84e3f0f9f566b8c1f1beaa34bdb118a5fe960f05`
- range manifest:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310.manifest.json`
  - sha256: `43ce140109e3ed97931c06a36b431919ad389dbed3e9c71818ab89c3a5dea8c5`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/manifest.md`
  - sha256: `1d158272963d5b9b0e32d4c4eba13c68b18dc0c3e2dae8a175992af5cde64cf7`
- source-contrastive target set:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json`
  - sha256: `9aa4a45892bce32b566232340f450749b9a074a8ce2c817c6de8901be15b1b08`
- surface ranking:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_headroom_surfaces.json`
  - sha256: `258512e9b3529bb5312cebd66097aee107c0ee5374376b56ba5715d520fc7e2b`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_materialize_jsonl_range.py tests/test_analyze_svamp_source_sidecar_cv_router_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/materialize_jsonl_range.py
```
