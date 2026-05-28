# Phase 9 M-HYST Preregistration: Hysteretic Protected-Set Updates

Date: 2026-05-28

## Status

Authorized by the unified sensitivity+budget+hysteresis sprint. M-HYST runs
after M-LAMBDA and only uses M-LAMBDA if the waterfilling ablation is retained
by its preregistered decision rule.

## Hypothesis

Hard position switching failed because discontinuous protected-set changes can
be worse than random controls. M-HYST tests whether entry/exit hysteresis
reduces boundary harm while preserving M11b's smooth EMA signal.

For each layer, the candidate score is

```text
s_i(t) = EMA(x_i(t)^2) - lambda * 1[i notin P_{t-1}]
```

Equivalently, channels enter when they rank in the top `k` but exit only after
falling below top `k + m`. The hysteresis margin `m` is fixed from calibration
before scoring.

## Models

- Granite-4.0-H-Small.
- Nemotron-3-Nano-30B-A3B-BF16.

## Regimes

1. BF16 reference.
2. Static top-1% W4A16 baseline.
3. Best retained previous ablation: M11b top-10 or M-LAMBDA.
4. M-HYST using the retained budget allocation.
5. Random hysteresis matched-budget control.

## Metrics

Recovery uses the standard positive-static-gap definition. The packet also
reports:

- protected-set churn rate by layer;
- fraction of channels retained only because of hysteresis;
- M-HYST minus previous-ablation median recovery;
- M-HYST minus random hysteresis control median recovery.

## Decision Rules

- PASS_HYST: M-HYST beats the retained previous ablation by at least `0.05`
  median recovery on either model with CI lower bound greater than `0`.
- KEEP_FOR_STACK: M-HYST improves churn or CI width without reducing median
  recovery by more than `0.02`; WJAC may use it as a stabilizer.
- DROP_HYST: M-HYST is worse than the previous ablation by more than `0.02`,
  or loses to random hysteresis control. M-WJAC then runs without hysteresis.
- FAIL_INFRA_HYST: invalid churn accounting, budget drift, or scoring failure.

## Runtime and Budget

Estimated cost: 3-4 GPU hours.

## Forbidden Actions

- Do not tune the hysteresis margin after scoring.
- Do not retain hysteresis in M-WJAC if this ablation drops it.
- Do not reinterpret M2/M10 failures as solved unless M-HYST beats its matched
  control.
