# Source-Only Sidecar Hash-Control Replay

- date: `2026-04-27`
- status: `live_failed_branch_pruned`
- branch: 1-byte source numeric residue sidecar with `shorter_than_target_numeric` guard
- scale-up rung: strict small live/holdout replay

## Purpose

Re-audit the prior positive source-only sidecar row with two stricter
requirements:

- target-side candidate pool only: `target` fallback plus `t2t`; no
  `source_alone` candidate artifact
- hash-based non-self shuffled-source and label-shuffle controls

The old textless SVAMP70 positive used `source_alone.jsonl` as a candidate pool
artifact, which allowed source-only values to appear directly in the decoder
pool. This replay tests whether the method still works when the source message
is only a residue sidecar and the decoder can select only target-side values.

## Live Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --source-quality-guard shorter_than_target_numeric \
  --shuffle-mode hash \
  --min-correct 26 \
  --min-target-self 0 \
  --min-clean-source-necessary 4 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 61 \
  --date 2026-04-27 \
  --output-json results/source_only_sidecar_hash_controls_20260427/live_shorter_guard_hash_controls.json \
  --output-md results/source_only_sidecar_hash_controls_20260427/live_shorter_guard_hash_controls.md \
  --output-predictions-jsonl results/source_only_sidecar_hash_controls_20260427/live_shorter_guard_hash_predictions.jsonl \
  --prediction-method source_shorter_than_target_guard_sidecar_live_hash
```

Live result:

- status: `source_only_sidecar_router_fails_gate`
- best matched correct: `22/70`
- clean matched: `0/6`
- clean source-necessary: `0/6`
- control clean union: `0/6`
- failing criteria: `min_correct`, `min_clean_source_necessary`

## Holdout Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --source-quality-guard shorter_than_target_numeric \
  --shuffle-mode hash \
  --min-correct 10 \
  --min-target-self 0 \
  --min-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 64 \
  --date 2026-04-27 \
  --output-json results/source_only_sidecar_hash_controls_20260427/holdout_shorter_guard_hash_controls.json \
  --output-md results/source_only_sidecar_hash_controls_20260427/holdout_shorter_guard_hash_controls.md \
  --output-predictions-jsonl results/source_only_sidecar_hash_controls_20260427/holdout_shorter_guard_hash_predictions.jsonl \
  --prediction-method source_shorter_than_target_guard_sidecar_holdout_hash
```

Holdout result:

- status: `source_only_sidecar_router_clears_gate`
- best matched correct: `11/70`
- clean matched: `1`
- clean source-necessary: `1`
- clean source-necessary ID: `daea537474de16ac`
- control clean union: `0`

## Decision

Kill the old source-only residue sidecar as a paper method. The holdout signal
is not enough because the stricter live replay fails once source-alone values
are excluded from the target-side candidate pool. The old live-positive row is
best treated as source-value leakage through the candidate pool, not a clean
communication method.

The remaining highest-value direction is source-surface discovery or a new
candidate-surface design that exposes enough target-side alternatives without
including source-only values.

## Hashes

- `live_shorter_guard_hash_controls.json`:
  `0e76eee72e35d3e7ee47904a5e0a47bdefaa7467b5b14dcbd1f4e11b9df9d0b5`
- `live_shorter_guard_hash_controls.md`:
  `13c154d8b55dd4ece60eec15331341d58f69edc0caa8fefce62bceaf61241545`
- `live_shorter_guard_hash_predictions.jsonl`:
  `23817d39cf5bbd7f9d65b75a8bbce95af5213b943ef28095c0373b9f5da326b1`
- `holdout_shorter_guard_hash_controls.json`:
  `0f2fd100c8989745073fb3b07661d8c30cbd69563a5b38e6abbf6c45af7417d8`
- `holdout_shorter_guard_hash_controls.md`:
  `48ebe194b2c5877557752d2a866bf2b5999ff360914804ac09aa702224b2d714`
- `holdout_shorter_guard_hash_predictions.jsonl`:
  `e14da8e6e7d85fb7b3ec969c0abbf681b0bf1a55ba4556aa2dec89d2b5bae037`
