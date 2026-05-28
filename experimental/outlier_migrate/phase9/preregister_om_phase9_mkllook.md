# Phase 9 M-KLLOOK Preregistration: Offline KL-Lookahead Oracle

Date: 2026-05-28

## Status

Authorized by the unified sensitivity+budget+hysteresis sprint. This run is an
offline diagnostic oracle, not a deployable method. It runs after V1
ParoQuant-on-Nemotron completes and before proxy methods such as M-LAMBDA,
M-HYST, or M-WJAC.

## Purpose

The oracle objective for protected channel selection is

```text
P*_t = TopK_{l,i} h_{l,i}(t) * x_{l,i}(t)^2
```

where `h` is downstream sensitivity and `x^2` is channel energy. M11b uses
`x^2` alone. M-KLLOOK estimates the deployability ceiling of channel-set
protection by measuring each candidate channel's direct forward-KL benefit:

```text
Delta_i(t) = KL(p_BF16 || p_Q_without_i) - KL(p_BF16 || p_Q_with_i)
```

Channels with largest `Delta_i(t)` form the protected set. This is too
expensive for deployment because it requires many counterfactual forwards, but
it answers whether better sensitivity proxies are worth pursuing.

## Models and Scope

Primary diagnostic packets:

- Granite-4.0-H-Small, one deterministic AIME-2025 packet.
- Nemotron-3-Nano-30B-A3B-BF16, one deterministic AIME-2025 packet.

The runner may subsample layers and candidate channels to fit the diagnostic
budget, but the subsampling plan must be fixed before any oracle recovery value
is inspected and written to `oracle_sampling_config.json`.

## Regimes

1. BF16 reference.
2. Static top-1% W4A16 baseline.
3. M11b top-10 baseline, reused from validated packets.
4. M-KLLOOK oracle top-10.
5. Random matched-budget top-10 control.

## Metrics

Recovery uses the same positive-static-gap definition as prior Phase 9 methods:

```text
1 - (ppl_regime - ppl_bf16) / (ppl_static_1pct - ppl_bf16)
```

The packet also reports:

- oracle median recovery;
- oracle minus M11b top-10 median recovery;
- oracle minus random matched-budget median recovery;
- sampled layer/channel coverage;
- average KL delta separation between selected and non-selected candidates.

## Decision Rules

- CEILING_HIGH_PROXY_BOTTLENECK: M-KLLOOK beats M11b top-10 by at least `0.10`
  median recovery on either model and beats random control by at least `0.15`.
  Interpretation: sensitivity matters, and cheap proxies remain worth testing.
- CEILING_CONFIRMED_LOW: M-KLLOOK is less than or equal to M11b top-10 plus
  `0.05` on both models, or fails to beat random matched-budget control.
  Interpretation: channel-set protection ceiling is low in these packets;
  regime-aware protocol framing should stop searching for a universal channel
  method.
- AMBIGUOUS_ORACLE: oracle improves over M11b but by less than `0.10`, or only
  one model has usable oracle coverage.
- FAIL_INFRA_MKLLOOK: missing packet artifacts, incomplete counterfactuals, or
  invalid KL computations.

Headline changes are not made from M-KLLOOK alone because it is an offline
oracle.

## Runtime and Budget

Estimated cost: 3-4 GPU hours.

The runner must stop early if counterfactual scoring exceeds the diagnostic cap
and report partial coverage rather than silently changing the scope.

## Forbidden Actions

- Do not report M-KLLOOK as deployable.
- Do not tune the sampled candidate set after seeing recovery.
- Do not use M-KLLOOK to choose a paper headline without a cheap proxy result.
- Do not start M-LAMBDA/M-HYST/M-WJAC on the GPU until the V1 GPU run has
  completed and the single-GPU resource is free.
