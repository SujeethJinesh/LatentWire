# M11b Nemotron Path C Framing Decision

Timestamp: `2026-05-25T07:45Z`

## Context

The original Nemotron M11b replication packet failed infrastructure
validation because the `static_1pct` baseline lacked corrected W4A16
quantization metadata. Path C reran only `static_1pct` with corrected
quantization and reused the validated BF16, M11b top-1/top-5/top-10,
and `static_top10` score caches.

Path C completed successfully in
`experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`.

## Infrastructure Outcome

The infrastructure issue is resolved.

- `artifact_complete`: `true`
- `excluded_tensors.by_regime.static_1pct` is present
- `static_1pct` quantized tensors: `6052`
- `static_1pct` excluded tensors: `76`
- Checker decision: `PASS_M11B_NEMOTRON_REPLICATES`
- Checker pass regime: `m11b_top10`

## Valid Numerical Result

All recovery values below use the corrected `static_1pct` baseline.

| Regime | Median recovery | CI95 low | CI95 high | Included traces |
|---|---:|---:|---:|---:|
| `m11b_top1` | `-0.10320524640512702` | `-0.5325671049426282` | `0.1631772118823916` | `10/12` |
| `m11b_top5` | `0.4567361832703779` | `0.34588779238757594` | `0.794986745913201` | `10/12` |
| `m11b_top10` | `0.8147397989034302` | `0.2544392881776635` | `0.9225522648803115` | `10/12` |
| `static_top10` | `0.5943837436179227` | `0.3165935076461148` | `0.8144650015682393` | `10/12` |

No-recoverable-static-gap traces: `2/12` (`0.16666666666666666`).

Relevant margins:

- `m11b_top5 - static_top10`: `-0.1376475603475448`
- `m11b_top10 - static_top10`: `0.2203560552855075`
- `m11b_top5 - m11b_top1`: `0.559941429675505`
- `m11b_top10 - m11b_top1`: `0.9179450453085571`

## Framing Decision

Use the human's deterministic paper-framing protocol rather than the
checker label alone.

The checker marks the packet as `PASS_M11B_NEMOTRON_REPLICATES`
because `m11b_top10` satisfies the budget-scaling pass rule. However,
the preauthorized paper-framing rule was specified around the top-5
M11b configuration:

- top-5 median recovery is positive and above `0.30`
- top-5 CI lower bound is above `0.10`
- top-5 does **not** beat `static_top10` by `0.15+`

Therefore, for paper framing, this is:

`AMBIGUOUS_M11B_NEMOTRON_BUDGET_SIGNAL`

This is not a failed replication in the earlier sign-flip sense. It is
a valid cross-model positive budget signal whose best Nemotron point is
top-10 rather than Granite's top-5, and whose top-5 variant does not
beat the matched static budget control on Nemotron.

## Paper Consequence

Use the hedged-positive title direction unless later committee review
forces a narrower frame:

`Decode-Position Channel Drift at Long-Decode Reasoning: Budget Tuning as a Partial Cross-Model Remedy`

Paper language should state:

- M11b shows cross-model budget sensitivity with positive recovery on
  Granite and Nemotron.
- The optimal budget is not stable across models: Granite's strongest
  preregistered signal was top-5, while Nemotron's checker pass is top-10.
- On Nemotron, `static_top10` is already strong, so the top-5 M11b claim
  does not transfer as a clean method win over matched static protection.
- This supports "budget is load-bearing" but does not support a settled,
  model-invariant top-5 EMA method claim.

## Next Action

Proceed to the single coherent paper integration pass with this framing,
then merge the citation, figure, and related-work branches as part of
that pass.
