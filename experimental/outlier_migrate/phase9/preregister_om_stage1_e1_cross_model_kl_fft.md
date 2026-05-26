# Stage 1 E1 Preregistration: Cross-Model KL and FFT Diagnostics

Date: 2026-05-26

## Purpose

Replicate the Granite-Small dense KL accumulation and spectral diagnostics on
the two smaller cross-model packets that currently lack dense trajectories:
DeepSeek-R1-Distill-Qwen-1.5B and Falcon-H1. Nemotron-3-Nano is explicitly
deferred in Stage 1 after the throughput probe projected manual dense decode
beyond the 15 GPU-hour cap. This tests whether the mechanism story is
model-specific or general across the measured non-Nemotron reasoning-model
families.

## Models

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, snapshot
  `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- `tiiuae/Falcon-H1-0.5B-Instruct`, snapshot
  `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368`

Each model uses the already validated deterministic 12-trace AIME-2025 slice.

## Measurement

For each model:

1. Generate the BF16 greedy target trace to 20,000 decode tokens.
2. Record BF16 reference log-probability vectors on the same dense KL grid used
   for the Granite KL experiment.
3. Capture BF16 layer-output activation magnitudes every 100 decode tokens from
   100 through 20,000 for FFT and autocorrelation analysis.
4. Build three protected-channel regimes from the captured activation
   trajectories:
   - `static_1pct`: top 1% at decode position 100.
   - `decdec_reactive_top1_proxy`: endpoint oracle top 1% at decode position
     10,000, matching the DecDEC proxy used in the Granite diagnostics.
   - `m11_alpha_0_5`: top-1% EMA-smoothed protection with alpha 0.5 and the
     same cap/selection rule as M11.
5. Compute full-vocabulary `KL(BF16 || Q)` at each dense KL grid position for
   the three quantized regimes.
6. Fit fixed functional forms to each trace-mean KL trajectory:
   linear, sublinear square-root, and power laws with exponents 1.1, 1.25, 1.5,
   and 2.0.
7. Compute FFT spectral entropy and autocorrelation length for the position-100
   top-1% channels in each layer.

## Decision Rule

- `PASS_E1_CROSS_MODEL_KL_FFT`: both measured models have square-root sublinear as
  the best KL fit for every quantized regime, median spectral entropy at least
  0.75, and median activation autocorrelation length between 50 and 200 tokens.
- `AMBIGUOUS_E1_PARTIAL_REPLICATION`: one model satisfies the above
  model-level criterion.
- `KILL_E1_MECHANISM_NOT_CROSS_MODEL`: no measured model satisfies the model-level
  criterion, or the KL trajectories are fundamentally superlinear on at least
  two models.
- `FAIL_INFRA_E1`: required artifacts are missing, malformed, or internally
  inconsistent.

If `KILL_E1_MECHANISM_NOT_CROSS_MODEL` fires, paper integration pauses for
human framing review.

## Budget

Estimated GPU cost: 15 GPU hours under the revised narrowed scope. The hard cumulative cap is 360 GPU hours.
Pause before additional GPU work if the ledger reaches 340 hours; stop all GPU
work at 355 hours unless the currently running command must finish to write a
valid artifact packet.
