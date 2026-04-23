# SVAMP32 Query Innovation Resampler Gate 0.15

Date: 2026-04-23

## Paper Status

Not ICLR-ready. The project is still blocked on a source-specific positive
method that survives matched versus zero/shuffled-source controls on the frozen
SVAMP32 C2C-teacher gate.

## Current Story

SVAMP32 exact IDs remain the strongest same-pair decision surface:

- target-alone: `8/32`
- C2C teacher: `16/32`
- C2C-only target-complementary wins: `10`

The live branch this turn was the existing learned
`bridge_ridge_qk_dynalign_query_innovation_resampler_replace` checkpoint. The
question was whether its previously promising GSM8K32 `0.15` residual gate
transfers to the frozen SVAMP32 teacher surface and remains source-specific
under controls.

## Exact Blocking Gap

We still need a learned connector that recovers C2C-only wins on the frozen
SVAMP32 IDs while zero-source and deterministic shuffled-source controls do not.
If the learned innovation-resampler fails that gate, it should be treated as a
mechanism probe, not as the paper method.

## What Ran

Checkpoint reused from the earlier GSM8K32 innovation-resampler run:

- `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`

Frozen SVAMP32 eval file:

- `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`

First, a tiny fixed-gate transfer sweep on the frozen SVAMP32 IDs:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
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
  --gate-values 0.10 0.15 0.25 \
  --methods rotalign \
  --prediction-output .debug/svamp32_query_innovation_resampler_gate_sweep_20260423/live_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Then, for the only live gate (`0.15`), exact controls and teacher-provenance:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
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
  --gate-mode fixed \
  --fixed-gate 0.15 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_zero_source.jsonl \
  --source-kv-control zero \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
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
  --gate-mode fixed \
  --fixed-gate 0.15 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_shuffled_source_salt1.jsonl \
  --source-prompt-control shuffle_examples \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

```bash
./venv_arm64/bin/python scripts/analyze_c2c_teacher_innovation.py \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --source source=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --source t2t=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --candidate matched=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --control zero_source=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_zero_source.jsonl,method=rotalign_kv \
  --control shuffled_source=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_shuffled_source_salt1.jsonl,method=rotalign_kv \
  --output-json results/svamp32_query_innovation_resampler_gate015_20260423/c2c_teacher_probe_gate015.json \
  --output-md results/svamp32_query_innovation_resampler_gate015_20260423/c2c_teacher_probe_gate015.md
```

## Evidence

### SVAMP32 Gate Sweep

| Gate | Correct | Status |
|---:|---:|---|
| `0.10` | `7/32` | below target |
| `0.15` | `9/32` | only live gate |
| `0.25` | `8/32` | target parity |

### Gate `0.15` Exact Controls

| Row | Correct | Wins vs target | Losses vs target | Teacher-only recovered | Numeric coverage |
|---|---:|---:|---:|---:|---:|
| matched | `9/32` | 2 | 1 | 1 | `32/32` |
| zero-source | `8/32` | 1 | 1 | 1 | `32/32` |
| shuffled-source | `9/32` | 2 | 1 | 1 | `31/32` |

Matched win/loss IDs versus target:

- wins: `3e8a5691f5443495`, `575d7e83d84c1e67`
- losses: `c042f0a2949ff8e6`

Teacher-provenance:

- the only C2C-only teacher win recovered by the matched row is
  `575d7e83d84c1e67`
- zero-source also recovers `575d7e83d84c1e67`
- shuffled-source also recovers `575d7e83d84c1e67`
- the other matched-only win, `3e8a5691f5443495`, is not a C2C-only teacher win
  at all

Probe gate result:

- `candidate_teacher_recovery_explained_by_controls`

## Decision

Alive:

- innovation-only bounded residual connectors remain alive as a mechanism class:
  this checkpoint can move SVAMP answers while staying near target accuracy

Weakened:

- the current fixed-gate learned query-innovation-resampler checkpoint is
  weakened on the exact SVAMP32 paper gate
- its only teacher-only recovery is fully explained by both zero-source and
  shuffled-source controls

Saturated:

- reusing the current GSM-tuned `0.15` checkpoint directly on SVAMP32 is
  saturated as a paper-grade positive-method claim

## Next Exact Gate

Do not promote this branch or widen it cross-family. The next exact gate should
be a control-discriminating innovation connector on the same frozen SVAMP32
surface, for example:

- a verifier-gated or contrastive innovation bottleneck that explicitly rejects
  wins retained under zero/shuffled source
- promotion only if matched-source recovers at least `4/10` C2C-only wins,
  reaches at least `11/32`, loses at most `1` target-correct ID, and controls
  recover at most `1` of the same matched teacher-only wins

## Artifacts

- `results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl`
- `results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_zero_source.jsonl`
- `results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_shuffled_source_salt1.jsonl`
- `results/svamp32_query_innovation_resampler_gate015_20260423/c2c_teacher_probe_gate015.json`
- `results/svamp32_query_innovation_resampler_gate015_20260423/c2c_teacher_probe_gate015.md`
