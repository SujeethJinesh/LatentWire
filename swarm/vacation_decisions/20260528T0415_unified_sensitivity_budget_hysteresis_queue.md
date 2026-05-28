# Unified Sensitivity+Budget+Hysteresis Queue

Date: 2026-05-28T04:15Z

## Decision

Adopt the unified oracle-derived positive-method family:

```text
P*_t = TopK_{l,i} h_{l,i}(t) * x_{l,i}(t)^2
s_{l,i}(t) = q_{l,i} * EMA(x_{l,i}(t)^2) - lambda * 1[i notin P_{l,t-1}]
```

This replaces the M-FISH-first plan. M-FISH gradient infrastructure remains
deferred because M-WJAC tests most of the sensitivity hypothesis with free
weight-column norms and no backward graph.

## Run Order

1. Finish active V1 ParoQuant-on-Nemotron.
2. Run M-KLLOOK offline KL-lookahead oracle.
3. Run M-LAMBDA layerwise budget waterfilling.
4. Run M-HYST hysteresis only on retained budget baseline.
5. Run M-WJAC on all four models using retained components only.
6. Run M-GATE on DeepSeek only if WJAC does not rescue DeepSeek.
7. Run M-ROUTE on Nemotron only if V1 shows M11b remains uniquely positive.

## Composition Discipline

Every addition must show marginal value. If LAMBDA or HYST does not improve or
stabilize the prior ablation, it is dropped from the WJAC stack. The E3
ParoQuant+M11b result showed that plausible components can be sub-additive.

## Decision Surface

- PASS: unified family beats M11b by at least 0.05 median recovery on Granite,
  DeepSeek, or Falcon with CI lower bound above zero.
- PASS_RESCUE: WJAC or a conditional follow-up rescues DeepSeek above 0.50 or
  Falcon above 0.30.
- CEILING_CONFIRMED: M-KLLOOK also recovers little, validating the
  regime-aware protocol and stopping the universal-channel-method search.
- KILL: family is less than or equal to M11b everywhere.

## Framing

Regime-aware framing is unchanged. The protocol remains prospective in design
and descriptively evaluated on the four-model study. The unified family adds
candidate actions for the protocol; it does not convert the paper into a
single-remedy claim unless the data supports that.

## Immediate Constraint

The GPU is currently occupied by V1. No additional GPU experiment starts until
V1 completes.
