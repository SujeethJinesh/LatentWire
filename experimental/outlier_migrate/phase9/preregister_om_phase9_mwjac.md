# Phase 9 M-WJAC Preregistration: Weight-Jacobian Sensitivity Proxy

Date: 2026-05-28

## Status

Authorized by the unified sensitivity+budget+hysteresis sprint. M-WJAC replaces
the M-FISH-first plan. No gradient-capture infrastructure is authorized for this
branch unless M-WJAC shows that sensitivity weighting helps but the weight-norm
proxy is too coarse.

## Hypothesis

The oracle objective protects channels by downstream sensitivity times channel
energy:

```text
P*_t = TopK_{l,i} h_{l,i}(t) * x_{l,i}(t)^2
```

M11b approximates this with `x_i(t)^2` alone. M-WJAC uses a cheap local
sensitivity proxy:

```text
q_{l,i} = ||W_{l,:,i}||_2^2
s_{l,i}(t) = q_{l,i} * EMA(x_{l,i}(t)^2)
```

The weight-column norm is the diagonal contribution of channel `i` to the next
linear map's output energy under a local Jacobian approximation. It is available
without backward passes and works on all four measured architectures.

M-WJAC may inherit M-LAMBDA and/or M-HYST only if those ablations are retained
by their own decision rules.

## Models

- Granite-4.0-H-Small.
- Nemotron-3-Nano-30B-A3B-BF16.
- DeepSeek-R1-Distill-Qwen-1.5B.
- Falcon-H1-0.5B-Instruct.

## Regimes

1. BF16 reference.
2. Static top-1% W4A16 baseline.
3. M11b top-10 baseline.
4. Best retained q=1 ablation from M-LAMBDA/M-HYST.
5. M-WJAC top-10 with retained budget/churn components.
6. WJAC-shuffled control: weight norms randomly permuted within layer.
7. Random matched-budget control.

## Metrics

Recovery uses the standard positive-static-gap definition. The packet also
reports:

- per-layer correlation between `||W_{:,i}||^2` and EMA channel energy;
- WJAC selected-channel overlap with M11b;
- M-WJAC minus M11b top-10 median recovery;
- M-WJAC minus retained q=1 ablation median recovery;
- M-WJAC minus WJAC-shuffled control median recovery.

## Decision Rules

- PASS_WJAC: M-WJAC beats M11b top-10 by at least `0.05` median recovery on at
  least one architecture where M11b underperforms, with CI lower bound greater
  than `0`.
- PASS_RESCUE: M-WJAC exceeds `0.50` median recovery on DeepSeek or `0.30` on
  Falcon, with CI lower bound greater than `0`.
- KEEP_PROTOCOL_AXIS: M-WJAC does not pass globally but wins over shuffled-WJAC
  control and improves one weak regime directionally. It becomes a
  sensitivity-axis diagnostic for the regime-aware protocol.
- KILL_WJAC: M-WJAC is less than or equal to M11b everywhere, or loses to the
  shuffled-WJAC control. Sensitivity via this proxy is not sufficient.
- FAIL_INFRA_WJAC: weight-column norms cannot be mapped to the activation
  channels being protected, or scoring fails.

## Runtime and Budget

Estimated cost: 4-5 GPU hours for all four models using cached activation
summaries and endpoint scoring.

## Prior-Art Differentiation

- AWQ uses activation-aware static weight saliency for weight quantization.
  M-WJAC is decode-time protected activation-channel selection under measured
  long-reasoning drift.
- ChanMix addresses mixed-precision KV-cache/channel mixing. M-WJAC targets
  W4A16 GEMM activation-channel protection and uses dynamic EMA state.
- Activation-sensitivity taxonomies motivate sensitivity criteria broadly.
  M-WJAC tests a specific dynamic long-decode proxy with matched controls.

## Forbidden Actions

- Do not build full Fisher/gradient infrastructure for this branch.
- Do not retain LAMBDA or HYST if their own ablations dropped them.
- Do not report a four-way stack without the marginal ablation table.
- Do not interpret shuffled-control failure as evidence for the WJAC proxy.
