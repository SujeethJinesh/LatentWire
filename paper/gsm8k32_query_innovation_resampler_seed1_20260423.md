# GSM8K32 Query Innovation Resampler Seed-1 Readout

Date: 2026-04-23

## Question

Can the guarded query-resampler become target-safe positive evidence by fitting
only the source-conditioned innovation over the target-side bridge prediction,
then applying a bounded additive residual instead of full KV replacement?

## Implementation

Added `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`.

- Calibration reuses the guarded query-resampler path but fits
  `target - base_bridge_prediction` as the query-module target.
- Runtime translation returns the fitted innovation tensor.
- Fusion applies a bounded additive residual:
  `target_kv + gate * bounded_innovation`, with per-row norm capped at `0.25`
  of the target-side KV norm.
- The branch is wired through calibration choices, dynamic alignment mixtures,
  query-feature collection, prediction-teacher collection, evaluation live
  query-head dispatch, and the GSM8K residual sweep runner.

## Commands

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_innovation_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_innovation_resampler_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423 \
  --seed 1

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_innovation_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_innovation_resampler_seed1_gate025_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423 \
  --seed 1 \
  --gate 0.25

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_innovation_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_innovation_resampler_seed1_gate015_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423 \
  --seed 1 \
  --gate 0.15
```

The `0.15` row produced a candidate-positive live row, so matched zero-source
and shuffled-source controls were run at the same fixed gate.

## Live Rows

| Gate | Correct | Pair vs target | Coverage | Empty | Checkpoint finite | Status |
|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 2/32 | 0/0/32 | 32/32 | 0 | yes | target parity |
| 0.25 | 2/32 | 1/1/30 | 32/32 | 0 | yes | moves answers but harms target |
| 0.15 | 3/32 | 1/0/31 | 32/32 | 0 | yes | live pass before controls |

Checkpoint health:

- checkpoint nonfinite values: `0`
- first bad key: `-`
- checkpoint max abs: `6416.1553`
- checkpoint SHA256:
  `b1f0cfa62c67ffcbdbce631c6cfd80df3240e132e252b0775aef355940a557b8`
- health JSON SHA256:
  `960b0194f71fa4222feb12f8e832efea14a4cd67388b7b95ec355478f296a790`

## Source Controls For Gate 0.15

| Row | Correct | Pair vs target | Pair vs live | Live-win retention | Coverage | Deranged | Target fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| live | 3/32 | 1/0/31 | - | - | 32/32 | false | 0/32 |
| zero_source | 2/32 | 1/1/30 | 0/1/31 | 1/1 | 32/32 | false | 0/32 |
| shuffled_source_salt1 | 3/32 | 1/0/31 | 0/0/32 | 1/1 | 32/32 | true | 4/32 |

`scripts/analyze_gsm8k_source_controls.py` status:
`source_controls_do_not_clear_gate`.

The only live candidate-only win is `cc90bec57e693936`. It is retained by both
zero-source and shuffled-source controls. The diagnostic helper also marks the
candidate numeric answer as identical to the text-to-text answer on that win;
source-alone is wrong on the same item.

## Artifact Hashes

- live gate 0.10 summary JSON:
  `1a96d5a12b6f5ef02f62704f6bc1acdb7db90d2c2749e5366958dea0aedbfc4b`
- live gate 0.25 summary JSON:
  `15d28d6c2bceceb3c3dbcc8ae560e95edab6cb9ea4ffaa1394d3b71e4895bb34`
- live gate 0.15 summary JSON:
  `09b74a6c6bbbdc797f920f3d89951cd43b32d791f1b17af73c17e6467447927e`
- live gate 0.15 predictions:
  `e5d9c00be65a19cc13028194fb17016480c25a04da5fc642bd1928ddbc555eb9`
- zero-source predictions:
  `22cc242653b5c2c0178d1bdb770c0d76e5624dd76f40c4d1381fbd09d0c3bc03`
- shuffled-source predictions:
  `cd3b61fe7565fd83d820fadcfc8b3274dc7542f940b2ff2697efb65d7554bf40`
- source-control JSON:
  `646164c7bc8b515631b981fa0c01de1de4cfa741d7893a29ca4b2b2ee2ff13b3`
- candidate diagnostics JSON:
  `8b14c313ec6ba8c7d40eed82f782adc5ae9a80e8ea7f21da847df3d22f613b04`

## Decision

Implementation succeeded and the branch is useful as a finite residual
communication surface, but the GSM8K32 seed-1 evidence does not support a
paper claim. Gate `0.15` produces a tiny live improvement, but the same win is
preserved under zero-source and shuffled-source controls. This demotes the row
to a control artifact rather than real source communication.

The innovation-resampler idea is weakened, not killed as an architecture: it
can move answers through a target-safe additive path. The missing ingredient is
a control-discriminating source-dependence gate or a different live lane whose
wins vanish under shuffled/zero source.

## Next Gate

Do not widen this row to GSM70 or cross-family. The next exact gate should be
one of:

- Return to the strongest existing real lane and run the next seed/frozen-slice
  source-control gate.
- Add a source-control-aware verifier/gate that rejects residual changes
  retained under zero/shuffled source, then rerun this exact GSM8K32 seed-1
  gate.
- Wire zero/shuffled controls into the residual sweep wrapper so future
  candidate rows cannot be promoted before control readouts exist.
