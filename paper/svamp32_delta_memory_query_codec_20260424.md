# SVAMP32 Delta-Memory Query Codec Readout

- Date: 2026-04-24
- Status: `no_matched_gate_candidate_for_controls`
- Gate: same-family Qwen2.5-0.5B-Instruct -> Qwen3-0.6B, SVAMP32 exact-ID matched slice
- Branch: query-innovation module memory with source K/V, target-prior K/V, source-minus-target delta K/V, and learned slots

## Question

Does an explicit target-prior delta memory channel recover clean source-only
residual IDs beyond the target-cache/self-repair floor?

Promotion threshold:

- exact ordered ID parity on `32/32`
- numeric extraction coverage `>=31/32`
- preserve or improve the target self-repair decision surface
- recover at least `2/6` clean residual IDs before running memory-mask controls

## Result

The branch did not clear the matched gate.

| Row | Correct | Clean residual | Teacher-only | Delta vs self-repair | Target losses | Status |
|---|---:|---:|---:|---:|---:|---|
| `rotalign_kv_gate_0.17` | 9/32 | 0/6 | 1 | -5 | 1 | `matched_candidate_below_clean_gate` |
| `rotalign_kv_gate_0.15` | 9/32 | 0/6 | 1 | -5 | 1 | `matched_candidate_below_clean_gate` |
| `rotalign_kv_gate_0.20` | 8/32 | 0/6 | 1 | -6 | 2 | `matched_candidate_below_clean_gate` |
| `rotalign_kv_gate_0.12` | 8/32 | 0/6 | 1 | -6 | 2 | `matched_candidate_below_clean_gate` |

The best rows tie the previous target-memory K-only top-line at `9/32` but
recover no clean residual IDs. The only teacher-only recovered ID is
`575d7e83d84c1e67`, which is not one of the clean residual targets. Because
the combined row failed `>=2/6`, runtime memory-mask controls were intentionally
not run.

## Implementation

Code added this turn:

- `TranslatorConfig.innovation_conditional_delta_memory`
- `TranslatorConfig.innovation_memory_control`
- fit-time delta rows `source_predicted_k/v - target_prior_k/v`
- runtime memory controls:
  - `combined`
  - `no_delta`
  - `source_only`
  - `target_only`
  - `delta_only`
  - `slots_only`
- evaluator override `--innovation-memory-control`
- tests that verify CLI parsing, config validation, target-cache forwarding,
  finite runtime controls, and exact runtime memory row selection

Focused verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_translator_core.py \
  tests/test_calibrate_and_ablation.py \
  tests/test_evaluate_helpers.py -q
```

Result: `333 passed in 4.23s`

## Commands

Calibration:

```bash
./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_delta_memory_query_codec_20260424/checkpoints/qwen25_to_qwen3_svamp32_deltamem_konly_query_codec_r16_bank16_seed1.pt \
  --bits 4 \
  --alignment grouped_subspace_transport \
  --quantization-correction bridge_ridge_qk_dynalign_query_innovation_resampler_replace \
  --quantization-correction-rank 16 \
  --bridge-bank-size 16 \
  --innovation-target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --innovation-positive-weight 16 \
  --innovation-default-weight 1.0 \
  --innovation-target-self-preserve-weight 16 \
  --innovation-value-loss-weight 0.0 \
  --innovation-conditional-delta-memory \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --seed 1
```

Matched gate:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/svamp32_delta_memory_query_codec_20260424/checkpoints/qwen25_to_qwen3_svamp32_deltamem_konly_query_codec_r16_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --gate-mode sweep \
  --gate-values 0.125 0.15 0.175 0.20 \
  --methods rotalign \
  --prediction-output .debug/svamp32_delta_memory_query_codec_20260424/preds/deltamem_konly_attention_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Clean-target analyzer:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_gate_sweep_clean_targets.py \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --candidate-jsonl .debug/svamp32_delta_memory_query_codec_20260424/preds/deltamem_konly_attention_gate_sweep.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_delta_memory_query_codec_20260424/deltamem_konly_attention_clean_targets.json \
  --output-md results/svamp32_delta_memory_query_codec_20260424/deltamem_konly_attention_clean_targets.md \
  --expected-n 32 \
  --min-clean-residual-recovered 2 \
  --target-self-correct 14
```

## Artifacts

- checkpoint:
  - `.debug/svamp32_delta_memory_query_codec_20260424/checkpoints/qwen25_to_qwen3_svamp32_deltamem_konly_query_codec_r16_bank16_seed1.pt`
  - sha256: `29ff93c6d7291fb9a4e00ac35a7ffa519c4d71c8bd4a38062c0d748baecf4ebb`
- matched sweep:
  - `.debug/svamp32_delta_memory_query_codec_20260424/preds/deltamem_konly_attention_gate_sweep.jsonl`
  - sha256: `01b3524fc887ef46ad0dc0ce86aa5cd145ce6f962d852e748f8472d7f7afc93a`
- readout:
  - `results/svamp32_delta_memory_query_codec_20260424/deltamem_konly_attention_clean_targets.json`
  - sha256: `8ab4d02b947369428bf49b74e658cd8aa9fd944eee916c0393d6e597a864158c`
  - `results/svamp32_delta_memory_query_codec_20260424/deltamem_konly_attention_clean_targets.md`
  - sha256: `9fed7cdbfb75bacc4bee0617ace866525ad9f96fc6e630246e3e93074d0fbde2`
- logs:
  - `.debug/svamp32_delta_memory_query_codec_20260424/logs/calibrate_deltamem_konly_seed1.log`
  - sha256: `6e08e914aa3340b1aee4081faae82267096e432a0a22fbe6d903d29be364e5da`
  - `.debug/svamp32_delta_memory_query_codec_20260424/logs/evaluate_deltamem_konly_attention_gate_sweep.log`
  - sha256: `a2c9a64134692f6fb98bbd90573e7caa323bf9f2f1f5c7c79b0b1f9384557d41`

## Hypothesis Update

- Weakened: explicit target-prior delta memory, as implemented here, is
  sufficient to extract clean source residual information on SVAMP32.
- Killed for now: this specific K-only delta-memory query-codec configuration
  as a same-pair gate candidate.
- Still alive: delta-memory infrastructure as a diagnostic/control surface if a
  stronger source-conditioned objective or verifier-routed branch produces a
  promoted combined row.
- Saturated: scalar gate sweeps, K-only value-loss suppression, target-only
  memory, and this unregularized delta-memory row construction on the current
  SVAMP32 exact-ID slice.

## Next Gate

Do not widen benchmarks from this row. The next live gate should either:

- make the objective explicitly source-discriminative before runtime controls
  matter, for example matched-vs-shuffled source contrast on the same residual
  target set, or
- pivot to a low-cost verifier-gated repair selector that preserves the
  `14/32` self-repair floor while testing whether any source-conditioned row
  gives actionable candidate repairs.

The next experiment must still use SVAMP32 exact-ID parity and must clear
`>=2/6` clean residual IDs before larger slices, seed repeats, or cross-family
claims.
