# Qwen2.5-Math -> Qwen3 SVAMP70 Holdout Finalish Source Guard

- date: `2026-04-26`
- status: `fails_holdout_source_control_gate`
- scale-up rung: medium holdout falsification
- live branch entering run: fixed source-quality guarded 1-byte source sidecar

## Question

The original SVAMP70 source-sidecar surface had a real medium positive, but
the decoded-length guard failed on the disjoint holdout. This run asks whether
the other fixed guard, `finalish_short_numeric`, generalizes better on the same
holdout.

## Command

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --source-quality-guard finalish_short_numeric \
  --min-correct 10 \
  --min-target-self 0 \
  --min-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 64 \
  --output-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.json \
  --output-md results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.md \
  --output-predictions-jsonl results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_predictions.jsonl \
  --prediction-method source_finalish_guard_sidecar
```

## Result

Holdout baselines:

- target-alone: `8/70`
- source-alone: `8/70`
- text relay: `18/70`
- C2C: `37/70`
- target/source oracle: `14/70`
- clean source-only after text exclusion: `2`

Finalish guard sweep:

| Moduli | Bytes | Matched | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs |
|---|---:|---:|---:|---:|---:|---|
| `2,3` | 1 | `8/70` | `1` | `0` | `2` | none |
| `2,3,5` | 1 | `9/70` | `1` | `0` | `2` | none |
| `2,3,5,7` | 1 | `9/70` | `1` | `0` | `2` | none |
| `97` | 1 | `9/70` | `1` | `0` | `2` | none |

The source-destroying/control clean union is
`ab1e71e8928661d0`, `daea537474de16ac`, so the single matched clean recovery is
not source-necessary.

## Decision

Kill fixed source-quality guarded source-sidecar policies as the live branch.
The original SVAMP70 positive row was real on its surface, but both fixed
textless guards now fail the disjoint holdout:

- decoded-length guard: `10/70`, `0/2` clean source-necessary, `2/2` control
  leakage
- finalish-short-numeric guard: `9/70`, `0/2` clean source-necessary, `2/2`
  control leakage

Do not tune thresholds or moduli on this fixed-guard family. The next
highest-value branch is either a cross-validated source router with richer
features and explicit holdout validation, or a new source surface with more
stable source-only headroom.

## Artifacts

- surface scan:
  - `results/source_headroom_surface_scan_20260426/scan.json`
  - sha256: `9611574620e91181a029e1b60165555bba8234ebbb02fcb78748d7ced52b4a6b`
  - `results/source_headroom_surface_scan_20260426/scan.md`
  - sha256: `421f4bdf2a90c636e41da4f90f05c5aac0fa49bea5a5c21f28ceac0c64755afd`
- holdout finalish guard JSON:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.json`
  - sha256: `dc5b99e4500e414dae02241e7472734ee9aef51772cd55d9de9149c6c4dd9c1d`
- holdout finalish guard readout:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.md`
  - sha256: `d5b9c88a414ae71d796d8f742724d14ba5ce22ab92d0b0869af9676fcbc5fcd4`
- predictions:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_predictions.jsonl`
  - sha256: `a0b7d2336c515b38c1e053fea09d94fb39e6fc224390e2500bf067928652e45a`

## Next Exact Gate

Run a bounded cross-validated source-router design only if it has a new
feature family beyond the already-failed shallow length/numeric guards, and
freeze a holdout gate before looking at holdout labels. Otherwise switch to
source-surface discovery rather than tuning the current fixed sidecar family.
