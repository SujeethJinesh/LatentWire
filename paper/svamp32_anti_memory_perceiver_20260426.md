# SVAMP32 Anti-Memory Perceiver Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable source-necessary method plus larger-slice,
  seed-repeat, source-control, and cross-family gates
- current story: C2C and compact syndrome bounds show real headroom, but the
  learned Perceiver connector still learns target/control artifacts
- blocker: matched source does not beat target-only, slots-only, zero-source,
  or shuffled-source controls on clean residual IDs

## Gate

This cycle tested the smallest objective-level rescue for the failed Perceiver
answer-teacher branch: keep the previous zero/shuffle source controls and add
training-time anti-memory controls for `target_only` and `slots_only`.

Promotion required:

- matched-positive clean IDs at least `2/6`
- matched-only clean IDs at least `2/6`
- control-leak clean IDs `0/6`
- mean matched-control delta greater than `0`
- no generation unless the teacher-forced pre-gate passes

## Commands

Calibration:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_anti_memory_perceiver_20260426/checkpoints/qwen25_to_qwen3_svamp32_anti_memory_w080_ctrl050_am050_r16_b16_seed1.pt \
  --bits 4 \
  --alignment grouped_subspace_transport \
  --quantization-correction bridge_ridge_qk_dynalign_query_innovation_resampler_replace \
  --quantization-correction-rank 16 \
  --bridge-bank-size 16 \
  --innovation-connector-mode perceiver_queries \
  --innovation-target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --innovation-positive-weight 16 \
  --innovation-default-weight 1.0 \
  --innovation-target-self-preserve-weight 16 \
  --innovation-answer-teacher-weight 0.8 \
  --innovation-value-loss-weight 0.0 \
  --innovation-conditional-delta-memory \
  --innovation-control-weight 0.50 \
  --innovation-control-mode zero_and_shuffle \
  --innovation-contrastive-margin 0.001 \
  --innovation-anti-memory-control-weight 0.50 \
  --innovation-anti-memory-control-mode target_and_slots \
  --innovation-anti-memory-contrastive-margin 0.001 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --seed 1
```

Teacher-forced diagnostic:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py \
  --translator .debug/svamp32_anti_memory_perceiver_20260426/checkpoints/qwen25_to_qwen3_svamp32_anti_memory_w080_ctrl050_am050_r16_b16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --device mps \
  --fixed-gate 0.15 \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1 \
  --score-target-self
```

The same diagnostic was repeated at fixed gates `0.125`, `0.15`, and `0.20`.

## Evidence

Calibration:

- dynamic mixture samples: `1411`
- answer-teacher injected prompts: `6`
- answer-teacher injected samples: `277`
- average K alignment cosine: `0.951`
- average V alignment cosine: `0.734`
- checkpoint sha256:
  `6a3932946c6fcb580a1b136e1e5d710e555884a73c94def8ef1485fc613692ad`

Teacher-forced gates:

| Gate | Matched Positive Clean | Matched-Only Clean | Control Leak Clean | Mean Matched-Control Delta |
|---:|---:|---:|---:|---:|
| `0.125` | 2/6 | 0/6 | 2/6 | -0.8898 |
| `0.150` | 2/6 | 0/6 | 2/6 | -0.8921 |
| `0.200` | 2/6 | 0/6 | 2/6 | -0.8660 |

The positive clean IDs were still control-explained:

- `aee922049c757331`: zero-source beats matched
- `e3ab8666238a289e`: slots-only beats matched

## Decision

Kill this objective variant for the current branch. The new anti-memory loss
changed the training objective but did not change the decisive teacher-forced
failure mode: matched source never beats the controls on a positive clean ID.

## Next Gate

Promote `svamp32_source_only_sidecar_router_gate` as the next highest-value
branch. The source signal formation should be source-only, with target-side
candidate pools used only as decoder side information. Gate on the frozen
SVAMP32 `6 + 3` clean/preserve IDs before any medium or generation expansion.
