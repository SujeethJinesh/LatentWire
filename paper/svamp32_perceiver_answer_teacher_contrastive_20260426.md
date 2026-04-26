# SVAMP32 Perceiver Answer-Teacher Contrastive Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one stable deployable positive method plus medium/large,
  seed-repeat, source-control, and cross-family gates
- current story: target self-repair remains the strong decoder floor, C2C
  exposes headroom, but this receiver-conditioned answer-teacher connector did
  not produce source-necessary clean margins
- blocker: clean C2C residual IDs are still better explained by target/control
  information than by matched source communication

## Gate

Train the existing query-innovation resampler in its strongest available cheap
configuration:

- `perceiver_queries` receiver-conditioned connector
- gold answer-token teacher blend on clean residual prompts
- target-self preservation weight
- source-control contrast against zero and shuffled source
- conditional delta memory

Run only the teacher-forced matched-vs-control diagnostic first. Do not spend
generation compute unless matched source has positive margins over all controls
on at least `2/6` clean residual IDs.

## Commands

Calibration:

```bash
./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp32_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt \
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
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --seed 1
```

Teacher-forced diagnostic was run at gates `0.125`, `0.15`, and `0.20` using
`scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py` with
`--score-target-self`.

## Artifacts

- results manifest:
  - `results/svamp32_perceiver_answer_teacher_contrastive_20260426/manifest.md`
- checkpoint:
  - `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp32_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
  - sha256: `65aea1fc6db7e96d5a0df5e3d98380fe44549a3a2eb35dff4bc7c09a1d89a485`
  - not tracked, size `1.8G`
- calibration log:
  - `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/logs/calibrate_w080_ctrl050_seed1.log`
  - sha256: `dcee16b600a8918fc7fadcd433baf27dabe8ad9ef89586146daaeb6bce737101`
- teacher-forced JSON:
  - `teacher_forced_gate0125.json`
    - sha256: `db0641fb41a2e49106fd7a63c72b2c09f97d3946969c03df3598c201cb49435f`
  - `teacher_forced_gate015.json`
    - sha256: `7ce7a255e8f847c43caccfd98ed5f37131a515e023c6660d93b14b1b485f82c5`
  - `teacher_forced_gate020.json`
    - sha256: `9aed1c804d854e4193281b0e24df71407f465dd1fc647c044e79a8f0db6a8802`

## Evidence

| Gate | Status | Matched Positive Clean | Matched-Only Clean | Control Leak Clean | Mean Matched-Control Delta |
|---:|---|---:|---:|---:|---:|
| `0.125` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -1.1011 |
| `0.150` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -1.1543 |
| `0.200` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -1.2968 |

At gate `0.15`, the two positive clean margins were not source-necessary:

- `aee922049c757331`: matched `14.5045`, shuffled-source `14.8191`
- `e3ab8666238a289e`: matched `5.3924`, target-only `5.7308`

Target-self rows were not preserved in the teacher-forced margin diagnostic.

## Decision

Do not run generation for this checkpoint. The teacher-forced pre-gate fails
across all tested fixed gates.

Weaken the Perceiver answer-teacher plus contrastive delta-memory branch. This
is now the second nearby learned-query/delta-memory variant to fail for the
same reason: controls or target-only memory explain the apparent clean-ID
signal.

## Next Gate

Move to the freshest surface with stronger measured clean source-only headroom
before spending on another connector variant. The newly materialized SVAMP70
C2C-vs-process-repair target set has `10` clean C2C source-only IDs after
excluding the process-repair baseline and full numeric coverage. Before
training there, materialize matching source-destroying controls or choose an
objective that can be evaluated with target-only/slots-only controls at
teacher-forced time.
