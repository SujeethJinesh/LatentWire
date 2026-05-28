# Phase 9 M-PRED Extension: DeepSeek and Falcon

Date: 2026-05-28

## Status

This addendum is authorized after the Granite M-PRED reduced run returned
`KILL_MPRED`. It does not alter the original Granite/Nemotron preregistration.
It tests whether the theory-matched M-PRED arm fills architectures where M11b
top-10 was ambiguous or weak.

## Scope

Models:

- DeepSeek-R1-Distill-Qwen-1.5B
- Falcon-H1-0.5B-Instruct

Both reuse validated V2 M11b packets for BF16, static top-1%, static top-10,
M11b top-5, and M11b top-10 baselines.

## Regime

The only new scored treatment arm is `mpred_top10_alpha_0_95`.

This is the high-information arm selected after Granite: the top-10 budget tests
positive-method headroom on architectures where M11b did not cleanly pass, and
`alpha=0.95` is the theory-matched value from the activation autocorrelation
analysis.

## Decision Rule

- PASS_ARCHITECTURE_FILL: M-PRED median recovery is above 0.30 and its 95% CI
  lower bound is above 0.0 on DeepSeek or Falcon.
- KILL_UNIFORM: M-PRED is no better than M11b on both extension architectures.
- AMBIGUOUS: M-PRED is positive but does not clear PASS_ARCHITECTURE_FILL.

Per-experiment cap: 8 GPU hours total for DeepSeek plus Falcon. If the cap
forces a partial run, DeepSeek is scored first and Falcon is deferred with an
explicit scope note.
