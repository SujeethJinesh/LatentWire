# Qwen2.5-Math -> Qwen3 SVAMP70 Chal241 Sidecar Gate

- date: `2026-04-26`
- status: `weak_surface_sidecar_gate_failed`
- scale-up rung: strict small / source-surface pruning
- current ICLR readiness: not ready

## Start Status

- current story: the Qwen2.5-Math source-contrastive sidecar has a real
  positive original SVAMP70 row, but the fixed and cross-validated guards failed
  on a disjoint holdout.
- exact blocker: determine whether the weak-but-clean `chal241-310` surface is
  worth C2C or connector spend.
- live branch: source-derived 1-byte numeric residue sidecar with source/target
  guards.
- highest-priority gate: beat target with at least `2/4` clean source-necessary
  IDs and zero clean control leakage.

## Surface

From the existing `chal241-310` source scout:

- target-alone: `10/70`
- source-alone: `5/70`
- text relay: `14/70`
- source-only over target: `4`
- clean source-only after text exclusion: `4`
- target/source oracle: `14/70`

## Commands

Text-relay agreement guard:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,method=source_alone \
  --candidate source=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --preserve-on-agreement-label t2t \
  --min-correct 12 \
  --min-target-self 0 \
  --min-clean-source-necessary 2 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 63 \
  --output-json results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_only_sidecar_router_t2t_guard.json \
  --output-md results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_only_sidecar_router_t2t_guard.md
```

Textless shorter-than-target guard:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,method=source_alone \
  --candidate source=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --source-quality-guard shorter_than_target_numeric \
  --source-quality-score-field source_target_len_ratio \
  --source-quality-max-threshold 1.0 \
  --min-correct 12 \
  --min-target-self 0 \
  --min-clean-source-necessary 2 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 63 \
  --output-json results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_shorter_than_target_guard_sidecar.json \
  --output-md results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_shorter_than_target_guard_sidecar.md \
  --output-predictions-jsonl results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_shorter_than_target_guard_predictions.jsonl \
  --prediction-method source_shorter_than_target_guard_sidecar
```

## Results

Text-relay agreement guard:

- best matched: `9/70`
- clean source-necessary: up to `3/4`
- control clean union: `1/4` to `2/4`
- failing criteria: below `12/70` minimum and nonzero clean control leakage

Textless shorter-than-target guard:

- best matched: `11/70`
- clean source-necessary: `1/4`
- control clean union: `1/4`
- failing criteria: below `12/70`, below `2/4` clean threshold, and nonzero
  clean control leakage

## Decision

Reject `chal241-310` as a sidecar/router decision surface. It is useful as a
weak diagnostic because the sidecar can touch clean source-only IDs, but source
signal is too sparse and controls leak clean IDs. Do not spend C2C generation
or connector training on this slice.

## Artifacts

- t2t-guard JSON:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_only_sidecar_router_t2t_guard.json`
  - sha256: `904942cceea20bc2e3e5f654c80532c19b5ce86b3e7fb998216c7ff0196f4ae8`
- t2t-guard markdown:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_only_sidecar_router_t2t_guard.md`
  - sha256: `df6e9a10e76023c6b1a5f701d9c87b915027bce884748a1f7d01b07e9bd577ff`
- textless guard JSON:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_shorter_than_target_guard_sidecar.json`
  - sha256: `34f6b87c2d25a67862f62b247cf2c2e69ace865ec452072e04273c9d0efc5b93`
- textless guard markdown:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_shorter_than_target_guard_sidecar.md`
  - sha256: `86e428e018e020aa915faff88a9ff238c673c41638c3cce907a445d3bbd714dd`
- textless predictions:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_shorter_than_target_guard_predictions.jsonl`
  - sha256: `d42f34613389fb254c9316e15f2b2af41393af9e77fb342ed9e1f6fe5486827e`

## Tests

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q`

## Next Exact Gate

Stop spending on adjacent SVAMP range scouting until a stronger source encoder
or different benchmark surface is identified. The next highest-value gate is a
source-surface scout on a different math/reasoning split where source-only over
target is at least `6/70` and clean source-only after text exclusion is at least
`4/70`; otherwise return to a learned communication objective with target-only
and slots-only anti-memory controls before any generation run.
