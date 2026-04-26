# GSM8K70 Seed4 Dynalign Source-Control Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: a stable source-derived positive method plus medium,
  large-slice, uncertainty, seed-repeat, and cross-family gates
- current story: GSM70 seed0 dynalign remains a useful mechanism clue, but
  finite repeat seeds do not preserve the lift
- blocker: raw `dynalign_module_replace_residrank16` is not seed-stable

## Gate

This gate revisited the strongest older benchmark clue after auditing the
older `paper/`, `rotalign`, `latent_bridge`, and `results/` evidence.

The selected branch was the raw GSM70
`dynalign_module_replace_residrank16` row because seed0 had `8/70` accuracy
against a `4/70` target baseline, and prior seed0 source controls retained
`0/6` live wins under zero/shuffled source with target fallback.

Seed4 was run from scratch with checkpoint health validation and integrated
source controls enabled. Source controls were configured, but the wrapper
correctly skipped them because the live row did not beat target.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 70 \
  --eval-file data/gsm8k_eval_70.jsonl \
  --materialized-eval-file results/gsm8k70_seed_repeat_full_20260422/_artifacts/gsm8k_eval_70.jsonl \
  --baseline-results-dir results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke \
  --results-dir .debug/gsm8k70_integrated_source_controls_20260426/seed4 \
  --checkpoints-dir checkpoints/gsm8k_contract_residual_sweep_20260421 \
  --seed 4 \
  --gate 0.10 \
  --run-source-controls \
  --source-control-random-salt 4 \
  --device mps \
  --dtype float32
```

## Evidence

| Seed | Accuracy | Correct | Pair vs target | Numeric coverage | Checkpoint health | Source controls |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0.1143 | 8/70 | 6W/2L/62T | 70/70 | finite | zero/shuffle retain 0/6 live wins |
| 3 | 0.0286 | 2/70 | 1W/3L/66T | 69/70 | finite | not run, live gate failed |
| 4 | 0.0571 | 4/70 | 3W/3L/64T | 70/70 | finite | not run, live gate failed |

Seed4 checkpoint health:

- nonfinite tensors: `0`
- first bad key: none
- max abs: `16326.4326`
- top tensor: `quant_aux_proj_V.12`
- checkpoint SHA256:
  `1d9e667fe90a7fbe4b06d982796d09f58398f18144780bb20f4950e774a0d26e`

## Decision

Kill raw GSM70 dynalign scale-up as the current live method branch. It remains
a mechanism clue because seed0 was source-control-supported, but seed4 shows
the failure is not just nonfinite checkpoint construction. A finite fresh seed
falls back to the target baseline and does not clear the live gate.

Do not spend large compute on more raw dynalign seed repeats unless a new
target-safe selection, routing, or source-derived objective changes the
hypothesis.

## Next Gate

Move the main line to source-derived sidecars with strict controls:

- learned source latent/token-feature predictors of C2C residue sidecars, or
- token/layer-level C2C residual distillation, or
- a query-bottleneck connector whose teacher-forced matched-only clean IDs beat
  zero/shuffled/target-only controls before generation.

Promotion rule remains:

- matched `>=14/32`
- target-self `3/3`
- clean source-necessary `>=2/6`
- numeric coverage `>=31/32`
- exact ordered ID parity
- zero-source, shuffled-source, label-shuffle, target-only, and slots-only
  clean union `0/6`

## Artifacts

See `results/gsm8k70_seed4_dynalign_source_controls_20260426/manifest.md`.
