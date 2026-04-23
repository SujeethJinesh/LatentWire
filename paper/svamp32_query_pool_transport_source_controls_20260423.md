# SVAMP32 Query Pool Transport Source Controls

Date: 2026-04-23

## Paper Status

Not ICLR-ready. The current same-pair story is still blocked on a connector that
recovers frozen-ID C2C-only wins on SVAMP32 while zero/shuffled-source controls
do not.

## Current Story

SVAMP32 exact IDs remain the strongest same-pair decision surface:

- target-alone: `8/32`
- C2C teacher: `16/32`
- C2C-only target-complementary wins: `10`

The live branch this turn was not a new model. It was a stricter runtime test
of the existing learned
`bridge_ridge_qk_dynalign_query_innovation_resampler_replace` checkpoint,
switching the selector from attention top-k to `query_pool_transport` on the
same frozen SVAMP32 slice.

## Exact Blocking Gap

We still need source-specific teacher-win recovery. If a runtime wrapper keeps
the same lone teacher-only recovery under zero or shuffled source, it is not a
paper method.

## Repo Fix

The exact shuffled-source probe initially failed with:

- `ValueError: query_pool_transport scores must have 73 positions, got 81`

Cause:

- shuffled source prompts can change source-token length
- runtime attention-derived position scores were not being resized to the
  translated KV length before `query_pool_transport` selection

Fix:

- added `_resize_position_scores()` in
  `latent_bridge/evaluate.py`
- routed runtime/source-attention score paths through that resize step before
  position selection
- added focused regression tests in
  `tests/test_evaluate_helpers.py`

Focused verification:

- `./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py -k 'query_pool_transport or resize_position_scores or shuffle_examples_uses_mismatched_source_prompt' -q`
- result: `5 passed`

## What Ran

Frozen SVAMP32 eval file:

- `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`

Checkpoint:

- `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`

Gate sweep used to choose the live row:

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
  --position-selection-metric query_pool_transport \
  --gate-mode sweep \
  --gate-values 0.10 0.15 0.25 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_query_pool_transport_20260423/live_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Exact matched/control rows for the live gate:

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
  --position-selection-metric query_pool_transport \
  --gate-mode fixed \
  --fixed-gate 0.10 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl \
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
  --position-selection-metric query_pool_transport \
  --gate-mode fixed \
  --fixed-gate 0.10 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_zero_source.jsonl \
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
  --position-selection-metric query_pool_transport \
  --gate-mode fixed \
  --fixed-gate 0.10 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_shuffled_source_salt1.jsonl \
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
  --candidate matched=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --control zero_source=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_zero_source.jsonl,method=rotalign_kv \
  --control shuffled_source=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_shuffled_source_salt1.jsonl,method=rotalign_kv \
  --output-json results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010.json \
  --output-md results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010.md
```

## Evidence

### Gate Sweep

| Gate | Correct | Status |
|---:|---:|---|
| `0.10` | `9/32` | only live row |
| `0.15` | `7/32` | below target |
| `0.25` | `6/32` | below target |

### Gate `0.10` Exact Controls

| Row | Correct | Wins vs target | Losses vs target | Teacher-only recovered | Numeric coverage |
|---|---:|---:|---:|---:|---:|
| matched | `9/32` | 2 | 1 | 1 | `32/32` |
| zero-source | `8/32` | 2 | 2 | 1 | `32/32` |
| shuffled-source | `9/32` | 2 | 1 | 1 | `31/32` |

Matched win/loss IDs versus target:

- wins: `3e8a5691f5443495`, `575d7e83d84c1e67`
- losses: `c042f0a2949ff8e6`

Control provenance:

- zero-source reproduces both matched wins
- shuffled-source reproduces both matched wins
- zero-source adds one extra target loss: `de2a795ab37694af`
- shuffled-source reproduces the full matched headline row despite one empty
  numeric extraction

Teacher-provenance:

- the only C2C-only teacher win recovered by the matched row is
  `575d7e83d84c1e67`
- zero-source also recovers `575d7e83d84c1e67`
- shuffled-source also recovers `575d7e83d84c1e67`
- source-alone also recovers `575d7e83d84c1e67`
- the other matched-only win, `3e8a5691f5443495`, is not a C2C-only teacher win

Probe gate result:

- `candidate_teacher_recovery_explained_by_controls`

### Target-Self-Repair Paper Gate

After adding the existing target-side repair row as a hard comparator, the
current live query-pool row fails the paper gate:

| Row | Correct | C2C-only recovered | Target losses | Note |
|---|---:|---:|---:|---|
| target_self_repair | `14/32` | `3/10` | `0` | target-only repair ceiling on this frozen slice |
| query_pool_matched | `9/32` | `1/10` | `1` | `-5` versus target_self_repair |

Gate verdict:

- `no_candidate_passes_target_self_repair_gate`

The matched row's only C2C-only recovery, `575d7e83d84c1e67`, is not a clean
source-communication win: it is unique versus target_self_repair, but retained
by both zero-source and shuffled-source controls.

Artifact sanity:

- exact ordered example-ID parity: `true` for target, matched, zero-source, and
  shuffled-source
- shuffled rows all carry `source_prompt_control=shuffle_examples`
- shuffled rows have `0` same-index source mappings

## Decision

Alive:

- decoder-conditioned innovation connectors are still alive as a method class
- the concrete next live idea is a target-conditioned innovation gate or ghost
  predictor that tries to suppress target-repair wins retained under controls

Weakened:

- `query_pool_transport` as a runtime wrapper on the current learned
  innovation-resampler checkpoint

Saturated:

- deterministic selector swaps on this checkpoint are saturated on the SVAMP32
  paper gate
- after the control-path bug fix, shuffled-source still reproduces the full
  matched row and the only teacher-only recovery

## Next Exact Gate

Do not widen this runtime family further. The next exact gate should be a
target-conditioned innovation connector on the same frozen SVAMP32 surface,
evaluated against:

- matched-source
- zero-source
- deterministic shuffled-source
- target-self-repair

Promotion threshold:

- matched-source reaches at least `16/32`
- matched-source beats target_self_repair by at least `+1`
- matched-source recovers at least `5/10` C2C-only wins
- at least `2` C2C-only wins are unique versus target_self_repair
- matched-source loses at most `1` target-correct ID
- zero-source and shuffled-source each retain at most `1` of the same matched
  teacher-only wins

## Artifacts

- `results/svamp32_query_innovation_query_pool_transport_20260423/live_gate_sweep.jsonl`
- `results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl`
- `results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_zero_source.jsonl`
- `results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_shuffled_source_salt1.jsonl`
- `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010.json`
- `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010.md`
- `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.json`
- `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.md`
- `results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_target_repair.json`
- `results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_target_repair.md`
