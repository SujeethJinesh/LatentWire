# Qwen2.5-Math SVAMP32 Perceiver C2C-Residual Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable source-necessary method plus medium,
  seed-repeat, strict source-control, and cross-family gates
- current story: Qwen2.5-Math -> Qwen3 exposes real C2C headroom on SVAMP32,
  but this target-conditioned Perceiver/query-innovation connector still does
  not transmit a source-necessary signal
- blocker: matched source margins remain explained by shuffled-source or
  slots-only controls on clean C2C residual IDs

## Gate

This cycle tested the smallest non-duplicative learned-connector smoke on the
current Qwen2.5-Math SVAMP32 C2C-headroom surface:

- source: `Qwen/Qwen2.5-Math-1.5B`
- target: `Qwen/Qwen3-0.6B`
- surface: `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`
- clean residual IDs: `6`
- target baseline: `8/32`
- C2C teacher: `15/32`
- connector: `perceiver_queries` query-innovation resampler
- training controls: zero-source, shuffled-source, target-only, slots-only
- pre-generation gate: matched source must beat all controls on at least `2/6`
  clean IDs

The first calibration attempt used `--innovation-target-self-preserve-weight
16` and failed because the compatible Qwen2.5-Math target set has no
`ids.target_self_repair` entries. The rerun set preserve weight to `0`; the
target-only-vs-C2C preservation IDs remain a separate generation/readout gate.

## Commands

Calibration:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --output .debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt \
  --bits 4 \
  --alignment grouped_subspace_transport \
  --quantization-correction bridge_ridge_qk_dynalign_query_innovation_resampler_replace \
  --quantization-correction-rank 16 \
  --bridge-bank-size 16 \
  --innovation-connector-mode perceiver_queries \
  --innovation-target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --innovation-positive-weight 16 \
  --innovation-default-weight 1.0 \
  --innovation-target-self-preserve-weight 0 \
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

Teacher-forced diagnostics were run with
`scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py` at gates
`0.125`, `0.15`, and `0.20` against matched, zero-source, shuffled-source,
target-only, and slots-only controls.

## Evidence

Calibration completed:

- dynamic token-mixture samples: `1411`
- answer-teacher injected samples: `274`
- average K alignment cosine: `0.961`
- average V alignment cosine: `0.799`
- checkpoint size: `1.9G`

Teacher-forced clean-ID results:

| Gate | Status | Matched Positive Clean | Matched-Only Clean | Control Leak Clean | Mean Matched-Control Delta |
|---:|---|---:|---:|---:|---:|
| `0.125` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -0.4836 |
| `0.150` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -0.4102 |
| `0.200` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -0.1916 |

At gate `0.15`, the two positive clean IDs were control-explained:

- `3e8a5691f5443495`: matched margin `4.0595`, shuffled-source margin
  `4.2457`
- `575d7e83d84c1e67`: matched margin `2.4345`, shuffled-source margin
  `3.2815`

## Artifacts

- checkpoint:
  - `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt`
  - sha256: `d50b00fd0b9f5b5afcb09af8f9ae89b868e913b0a0610ef8132e66f20c726759`
  - not tracked, size `1.9G`
- calibration log:
  - `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/logs/calibrate_seed1_preserve0.log`
  - sha256: `495a782080e78fc1b40f59452dc25ec936207f93cb1945dbce4063196002d156`
- teacher-forced JSON:
  - `results/qwen25math_svamp32_perceiver_c2c_residual_20260426/teacher_forced_gate0125.json`
  - sha256: `c60c357ecf38a76479a94265296ec3a32905bd95d4b39787c83da84429c21503`
  - `results/qwen25math_svamp32_perceiver_c2c_residual_20260426/teacher_forced_gate015.json`
  - sha256: `caf7de8d67de8fd06defc1bc71d68fa4c636877b05508f533fd1fe913cce690c`
  - `results/qwen25math_svamp32_perceiver_c2c_residual_20260426/teacher_forced_gate020.json`
  - sha256: `97d64048334ddc523b1a7303fb7e8922d46828c6d3bf3ee97e59445a81fc8eca`

## Decision

Kill this specific Qwen2.5-Math Perceiver/query-innovation checkpoint before
generation. It repeats the same decisive failure mode as earlier learned-query
connectors: positive clean margins exist, but matched source does not beat
source-destroying or memory controls.

Do not tune fixed gate, positive weight, answer-teacher weight, or anti-memory
weight on this exact architecture without a materially different source/target
conditioning path.

## Next Gate

The next live branch should implement a target-query-conditioned source
bottleneck rather than another receiver-conditioned memory variant:

- target hidden/query states query source states directly
- include target-only learned-prefix, slots-only prefix, shuffled-source, and
  zero-source controls at the same byte/query budget
- treat target cache/candidate pool as decoder side information and optimize
  only source innovation
- require at least `2/6` matched-only clean IDs before generation

## CPU Answer-Likelihood Follow-Up

The eval-only gold-answer continuation likelihood diagnostic found a small
positive clue, then killed the checkpoint on the full clean-ID gate.

Four-clean-ID smoke:

- result directory:
  `results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/`
- status: `answer_likelihood_controls_pass`
- matched mean answer logprob: `-7.989116`
- zero-source: `-8.250677`
- shuffled-source: `-8.131923`
- target-only: `-8.162249`
- slots-only: `-8.118848`
- best-control wins/losses/ties: `3/1/0`
- mean live-best delta: `+0.080362`

Six-clean-ID expansion:

- result directory:
  `results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/`
- status: `answer_likelihood_controls_fail`
- matched mean answer logprob: `-8.195434`
- zero-source: `-8.387585`
- shuffled-source: `-8.190414`
- target-only: `-8.192871`
- slots-only: `-8.191226`
- best-control wins/losses/ties: `4/2/0`
- mean live-best delta: `-0.090384`

Decision update: the 4-ID pass is only a partial mechanism clue. The checkpoint
is killed as a strict positive method candidate because the clean6 expansion
fails against shuffled-source, target-only, and slots-only controls on mean
answer likelihood.
