# SVAMP32 Answer-Teacher Microfit - 2026-04-24

## Status

- status: `no_teacher_forced_source_signal`
- readiness impact: negative for the current same-pair connector lane
- gate: matched-only positive answer margins on at least `2/6` clean residual IDs under zero-source, shuffled-source, target-only, and slots-only controls
- outcome: `0/6` matched-only clean residual IDs; no greedy generation sweep was justified

## Motivation

The previous teacher-forced diagnostic showed that the failed 8-query
Perceiver connector had no hidden source-specific answer signal. The promoted
next gate was a controlled microfit: inject direct answer-token supervision on
the six clean residual IDs, preserve target-self IDs, and test whether matched
source becomes necessary before spending compute on generation.

## Decision Surface

Top moves considered:

- Answer-token teacher microfit on the existing query-innovation connector.
  It matters because it is the cheapest way to test whether direct answer-side
  supervision can make the current connector source-dependent. It might fail by
  learning target-cache or learned-slot shortcuts. It costs one SVAMP32
  calibration plus one diagnostic and helps same-pair, interpretability, and
  reproducibility.
- Standalone differentiable answer-margin sidecar. It is the cleanest future
  implementation because the loss can directly optimize matched-vs-control
  margins. It might fail by overfitting six IDs and is a larger code change
  than the current turn needs.
- Latent syndrome sidecar. It tests whether source residual states encode
  numeric answer checksums rather than full cache transport. It might fail
  because the source/text union already recovers `0/6` clean residual IDs. It
  helps interpretability and efficiency, but is a new branch rather than the
  promoted gate.

I executed the answer-token teacher microfit because it directly clears or
kills the promoted gate using the existing translator and diagnostic surface.

## Implementation

Added default-off calibration support:

- `--innovation-answer-teacher-weight`
- `--innovation-answer-teacher-template`
- `inject_answer_token_teacher(...)`

The helper replaces prediction-teacher rows only for prompts whose stable IDs
match `ids.clean_residual_targets`. Target-self IDs still use the existing
zero-residual preserve mask. This keeps the microfit bounded to clean residual
answer supervision without changing the evaluator.

## Commands

Calibration:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt \
  --bits 4 \
  --alignment grouped_subspace_transport \
  --quantization-correction bridge_ridge_qk_dynalign_query_innovation_resampler_replace \
  --quantization-correction-rank 16 \
  --bridge-bank-size 8 \
  --innovation-connector-mode perceiver_queries \
  --innovation-target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --innovation-positive-weight 128 \
  --innovation-default-weight 1.0 \
  --innovation-target-self-preserve-weight 32 \
  --innovation-answer-teacher-weight 0.90 \
  --innovation-answer-teacher-template ' {answer}' \
  --innovation-value-loss-weight 1.0 \
  --innovation-conditional-delta-memory \
  --innovation-control-weight 0.30 \
  --innovation-control-mode zero_and_shuffle \
  --innovation-contrastive-margin 0.010 \
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
  --translator .debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_answer_teacher_microfit_20260424/answer_teacher_w090_gate015_clean_self.json \
  --output-md results/svamp32_answer_teacher_microfit_20260424/answer_teacher_w090_gate015_clean_self.md \
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

## Evidence

Calibration:

- dynamic token-mixture samples: `1411`
- clean residual prompts matched: `6`
- answer-teacher samples injected: `277`
- target-self-preserve prompts matched: `3`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- checkpoint: `.debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt`

Diagnostic summary:

- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.834308`
- mean best-control margin: `-2.637356`
- mean matched-minus-control margin: `-1.196952`

Clean residual rows:

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| `13cb77b698eeadb5` | 8142 | 46 | -13.573610 | -12.979069 | -0.594540 | shuffled_source | control_or_negative |
| `1d50b408c8f5cd2c` | 949 | 1 | -11.451377 | -11.165769 | -0.285608 | zero_source | control_or_negative |
| `2de1549556000830` | 39 | 33 | -9.640119 | -9.043700 | -0.596419 | zero_source | control_or_negative |
| `6e9745b37ab6fc45` | 61 | 600 | -7.988647 | -2.830618 | -5.158029 | shuffled_source | control_or_negative |
| `aee922049c757331` | 1 | 17 | 14.211058 | 14.415606 | -0.204548 | slots_only | control_or_negative |
| `e3ab8666238a289e` | 1 | 4 | 5.436848 | 5.779413 | -0.342566 | target_only | control_or_negative |

The answer-token teacher did not produce source-specific transfer. The same
two easy positive clean IDs remain control-driven; the strongest controls are
now `slots_only` and `target_only`, which is a learned-slot or target-prior
shortcut rather than source communication.

## Artifacts

- `results/svamp32_answer_teacher_microfit_20260424/answer_teacher_w090_gate015_clean_self.json`
- sha256: `e9db6ffed6ba5c42a9b983e48154fde3eac98248c56b05c57900cd9870266f71`
- `results/svamp32_answer_teacher_microfit_20260424/answer_teacher_w090_gate015_clean_self.md`
- sha256: `324e123639f812030b3e5e3f8c1ab81127468010f0438c3ee5752ca166a1a6e2`
- `.debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt`
- sha256: `437b7eecf8f0b3704eb8e6260cefcd9d45ead2a31d02855c33655c06dd2de8fc`
- `.debug/svamp32_answer_teacher_microfit_20260424/logs/calibrate_answer_teacher_w090_r16_q8_seed1.log`
- sha256: `8cbfe57de7c83d86fbae9c46e134f08110938794a6b2f60606456cb9b4091d88`
- `.debug/svamp32_answer_teacher_microfit_20260424/logs/diagnostic_answer_teacher_w090_gate015_clean_self.log`
- sha256: `ada211e52f4d0b3189a5a5ce2d9487536367419d0db5705942fdb0c9302461a1`

## Subagent Synthesis

- literature and internet agents converged on C2C-distilled learned-query
  fusers, using Q-Former/Perceiver-style bottlenecks and source-destroying
  controls
- ablation agent recommended no greedy sweep unless the teacher-forced
  matched/control gate cleared
- repo-audit agent recommended a standalone differentiable answer-margin
  microfit as the cleaner next implementation if this calibration proxy failed
- creative agent suggested a latent syndrome sidecar as a non-cache branch:
  transmit interpretable numeric residues with target candidates as decoder
  side information

## Hypothesis Update

- killed: calibration-time answer-token teacher injection is enough to make the
  current query-innovation Perceiver connector source-dependent
- killed: running greedy generation from this microfit is evidence-driven
- weakened: same-pair Qwen2.5-0.5B to Qwen3-0.6B contains easily extractable
  source-specific residual answer signal in this connector family
- still alive: a fully differentiable answer-margin sidecar that directly
  optimizes matched-vs-control margins
- promoted: before any more same-pair connector tuning, run a source
  informativeness gate or switch to the standalone margin sidecar / latent
  syndrome sidecar as an explicit new branch

## Next Exact Gate

Do not run another calibration-proxy microfit for this connector. The next gate
should be one of:

- source-informativeness audit on the six clean residual IDs, including source
  answer margin and candidate-pool oracle coverage
- standalone differentiable answer-margin sidecar, promoted only if it clears
  the same matched-only `>=2/6` teacher-forced gate
- latent syndrome sidecar, promoted only if source-derived numeric residues
  beat zero/shuffle controls on the clean residual IDs
