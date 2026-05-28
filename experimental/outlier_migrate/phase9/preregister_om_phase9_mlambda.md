# Phase 9 M-LAMBDA Preregistration: Layerwise Budget Waterfilling

Date: 2026-05-28

## Status

Authorized by the unified sensitivity+budget+hysteresis sprint. M-LAMBDA is the
first deployable ablation after the M11b EMA baseline.

## Hypothesis

M11b showed that protected budget is a determining factor: top-1% EMA was weak,
while top-10% EMA passed on Nemotron. M-LAMBDA tests whether reallocating the
same total protected-channel budget across layers improves recovery over a flat
per-layer budget.

The scoring axis remains M11b's `q=1` EMA over squared activation magnitude.
The only new component is layerwise waterfilling:

```text
k_l = arg greedy waterfill_l Delta_l(k)
Delta_l(k) = R_l(k) - R_l(k-1)
```

where `Delta_l(k)` is estimated from fixed calibration marginal curves. The
total budget is held equal to the M11b top-10 budget unless explicitly reported
as a separate sensitivity arm.

## Models

- Granite-4.0-H-Small.
- Nemotron-3-Nano-30B-A3B-BF16.

These two models are the current positive-regime anchors: ParoQuant is strong
on Granite, and M11b top-10 is strong on Nemotron.

## Regimes

1. BF16 reference.
2. Static top-1% W4A16 baseline.
3. M11b top-10 flat per-layer budget baseline.
4. M-LAMBDA top-10 total budget, waterfilled by layer.
5. Random layer-waterfilled matched-budget control.

## Metrics

Recovery uses the standard positive-static-gap definition. The packet also
reports:

- layer budget allocation `k_l`;
- marginal calibration curves by layer;
- M-LAMBDA minus M11b top-10 median recovery;
- M-LAMBDA minus random control median recovery;
- no-gap fraction.

## Decision Rules

- PASS_LAMBDA: M-LAMBDA beats M11b top-10 by at least `0.05` median recovery
  on either model with CI lower bound greater than `0`.
- KEEP_FOR_STACK: M-LAMBDA improves median recovery by `0.02-0.05` or narrows
  CI width materially without harming the other model. It may be retained for
  M-HYST/M-WJAC as a stabilizing component.
- DROP_LAMBDA: M-LAMBDA is less than or equal to M11b top-10 on both models, or
  loses to random layer-waterfilled control. Later ablations revert to flat
  budgets unless WJAC has an explicit reason to retest allocation.
- FAIL_INFRA_LAMBDA: missing marginal curves, invalid budget totals, or scoring
  failures.

## Runtime and Budget

Estimated cost: 4-5 GPU hours.

## Forbidden Actions

- Do not change total protected budget after inspecting recovery.
- Do not proceed as if waterfilling helped unless the marginal comparison
  supports it.
- Do not compose M-LAMBDA into later methods if it is explicitly dropped by
  the decision rule.
