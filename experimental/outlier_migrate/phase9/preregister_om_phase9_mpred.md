# Phase 9 M-PRED Preregistration: Predictive Channel Tracker

Date: 2026-05-27

## Status

This experiment is authorized after Stage 1 integration, citation additions,
math formalization, V1/V2 baseline vetting, and the M-PRED scoop check. The
scoop check found no direct 2025--2026 prior art for one-step predictive
protected-channel selection in W4A16 long-decode reasoning.

## Hypothesis

M11b uses a lagging EMA over observed high-magnitude channels. If some recovery
loss comes from that lag, a one-step predictive tracker should choose a better
endpoint protected set at fixed budget.

For channel `c` at update step `t`, M-PRED maintains

```text
m_hat_c(t+1) = alpha * m_hat_c(t) + (1 - alpha) * |x_c(t)| + beta_c * I_c(t)
I_c(t) = |x_c(t)| - m_hat_c(t-1)
```

where `beta_c = sigma_meas_c^2 / (sigma_meas_c^2 + sigma_proc_c^2)` is estimated
once from calibration activations. We test `alpha in {0.5, 0.8, 0.95, 0.99}`.
The high-alpha arms reflect the observed activation-spectrum autocorrelation of
roughly 100 decode tokens; the KL AR(1) coefficient is not used as a direct
channel-magnitude coefficient.

## Scope

Primary models:

- Granite-4.0-H-Small
- Nemotron-3-Nano-30B-A3B-BF16

The experiment uses the deterministic AIME-2025 12-trace slice and the existing
W4A16 endpoint scoring protocol. Because the current W4A16 runner applies
protected sets as static tensor exclusions during endpoint scoring, this
experiment evaluates M-PRED's predicted final protected set at position 10,000
for the 512-token endpoint window. A fully online per-token protection
implementation is future work.

If the 12-hour M-PRED cap prevents a full Granite+Nemotron run, Granite runs
first and Nemotron is deferred with an explicit scope note.

## Regimes

Baselines reused from validated M11b packets:

1. BF16 reference
2. static top-1%
3. M11b top-5
4. M11b top-10
5. static top-10

New scored regimes:

1. M-PRED top-5, alpha=0.5
2. M-PRED top-5, alpha=0.8
3. M-PRED top-5, alpha=0.95
4. M-PRED top-5, alpha=0.99
5. M-PRED top-10, alpha=0.95
6. Random-walk top-5 matched-budget control
7. M-PRED random-alpha top-5 control, alpha sampled per channel from
   Uniform[0.5, 0.99]

## Metrics

Per-trace recovery uses the existing positive-static-gap definition:

```text
1 - (ppl_regime - ppl_bf16) / (ppl_static_1pct - ppl_bf16)
```

Traces with no positive static top-1% recoverable gap are excluded from recovery
summaries and counted in the no-gap fraction.

## Decision Rules

Let `best_mpred` be the best median recovery among M-PRED arms, and `best_m11b`
be the better of M11b top-5 and M11b top-10.

- PASS: `best_mpred - best_m11b >= 0.05` on at least one model and the 95% CIs
  do not overlap.
- PASS_TIGHTENS_GRANITE: on Granite, best M-PRED has non-negative median and
  CI width at least 25% narrower than M11b top-5.
- AMBIGUOUS: M-PRED moves in a positive direction but CIs overlap M11b.
- KILL: M-PRED loses to M11b on the measured model, or loses to the random-alpha
  control.

If an individual M-PRED arm passes, composition experiments may be considered
later. No composition is run in this preregistered experiment.

## Runtime and Budget

Per-experiment cap: 12 GPU hours.

Cumulative GPU budget rules still apply:

- hard cap 360 hours
- stop threshold 340 hours

If the cap is reached mid-experiment, the runner commits the partial packet,
records which regimes completed, and the checker reports the valid subset.

## Prior-Art Check

Checked sources include DecDEC (arXiv:2412.20185), AttentionPredictor
(arXiv:2502.04077), Rotated Runtime Smooth (arXiv:2409.20361), ASER
(arXiv:2411.07762), and GuidedQuant (arXiv:2505.07004). None directly tests
Kalman/AR one-step-ahead channel protection for W4A16 long-decode reasoning.
