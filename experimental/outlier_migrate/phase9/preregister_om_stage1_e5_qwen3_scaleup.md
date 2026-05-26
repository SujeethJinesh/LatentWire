# Stage 1 E5 Preregistration: Qwen3-8B Partial Scale-Up

## Status

Frozen before any E5 Qwen3-8B row is inspected. This experiment is part of
the revised Stage 1 sequence and runs only after E3, E2, E1, E4, and E15/E6
ordering constraints permit it.

## Motivation

The current evidence base covers Granite-Small, Nemotron-3, DeepSeek-R1-
Distill-Qwen-1.5B, and Falcon-H1. E5 asks whether the channel-set drift
phenomenon and the M11b top-10 budget remedy survive on a larger modern
reasoning-family checkpoint. The revised scope is Qwen3-8B only because
Qwen3-14B and larger models are expected to consume too much of the remaining
GPU budget under the observed decode throughput.

## Model and Data

- Model: `RedHatAI/Qwen3-8B-quantized.w4a16`
- Prompt slice: deterministic AIME-2025 indices 0-11
- Trace count: 12
- Drift horizon: top-1% set at decode position 100 compared with position
  20000
- M11b scoring horizon: position 10000, matching the existing M11b protocol
- M11b update cadence: every 100 decode tokens through position 10000
- M11b alpha: 0.3
- M11b protected budget: top-10%
- Bootstrap seed: 20260526

No substitute Qwen checkpoint is permitted. If the RedHatAI checkpoint is
unavailable locally, does not load, lacks compatible dense weights after
decompression, or runs beyond the preregistered cap, the runner must emit a
`SKIPPED_INFRA_E5` or `FAIL_INFRA_E5` packet rather than silently switching
models or changing scope.

## Metrics

1. Strict top-1% set-leaving rate: for each layer, select the channels in the
   top 1% by mean absolute activation at decode position 100 and measure the
   fraction not still in the top 1% at decode position 20000. Aggregate across
   layers and traces.
2. M11b top-10 recovery:

   `1 - (ppl_m11b_top10 - ppl_bf16) / (ppl_static_1pct - ppl_bf16)`

   Traces with non-positive static top-1% gap are excluded from the recovery
   median and reported as no-gap traces.

## Decision Rule

- `PASS_E5_QWEN3_SCALEUP`: strict top-1% set-leaving rate is at least 0.40,
  M11b top-10 median recovery is greater than 0.30, and the bootstrap CI95
  lower bound is greater than 0.10.
- `AMBIGUOUS_E5_QWEN3_SCALEUP`: drift is present but the recovery criterion is
  mixed, confidence intervals are too wide, the run is incomplete but
  scientifically interpretable, or one of the two signals is present without
  the other.
- `KILL_E5_QWEN3_SCALEUP`: drift is absent or M11b top-10 recovery is absent.
- `SKIPPED_INFRA_E5`: the model cannot be loaded or decoded within the
  preregistered 12 GPU-hour cap.
- `FAIL_INFRA_E5`: artifacts are missing, malformed, or fail recomputation.

## Budget

Cap: 12 GPU hours. If Qwen3-8B throughput is itself problematic, document the
state and stop E5. Do not extend to Qwen3-14B under this preregistration.

## Non-Authorizations

- No Qwen3-14B, Qwen3-32B, or alternate checkpoint fallback.
- No alpha tuning.
- No budget sweep.
- No prompt-slice changes after observing results.
- No dropping the static top-1% denominator regime.
