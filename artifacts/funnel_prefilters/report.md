# Funnel Prefilters Artifactization

Date: 2026-05-28T04:14:34Z

## Status

- Paper readiness: not ICLR-ready; no positive method has yet survived the required larger frozen-slice, seed-stable, cross-family gate.
- Current story: HybridKernel remains the only active positive-method branch; these CPU artifacts only decide whether M11b-derived HYST/LAMBDA smoke work is worth a gated GPU request.
- Submission blocker: no benchmark-backed positive method with paired uncertainty and strict same-family/cross-family separation.

## Inputs

No GPU jobs were run. I read the run ledger/reviewer context plus cached M11b packets only.

| Model | Run directory |
|---|---|
| Granite | `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z` |
| Nemotron | `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z` |
| DeepSeek | `experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z` |
| Falcon | `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z` |

## HYST Gate

Decision: **RUN_HYST_SMOKE_ON_DEEPSEEK_FALCON**.

Gate used: run only when top-10 M11b protected-pool late adjacent churn is persistently high (`>=0.20` median) and the late adjacent high-score pool remains locally stable enough (`>=0.70` median stability).

| Model | Top10 first-final drift mean | Late adjacent churn median | Late pool stability median | Gate |
|---|---:|---:|---:|---|
| Granite | 0.676 | 0.238 | 0.762 | pass |
| Nemotron | 0.579 | 0.044 | 0.956 | fail |
| DeepSeek | 0.431 | 0.289 | 0.711 | pass |
| Falcon | 0.446 | 0.298 | 0.702 | pass |

Readout: DeepSeek and Falcon pass the smoke gate. Granite also passes these CPU metrics but is outside the requested DeepSeek/Falcon smoke surface; Nemotron is locally saturated and fails the persistent late-churn side of the gate.

## LAMBDA Gate

Decision: **RUN_LAMBDA_SMOKE_ON_DEEPSEEK_FALCON**.

Gate used: run only when layers differ meaningfully in endpoint drift and dominant-layer allocation, and the trace surface is not saturated by no-gap rows. This is a smoke gate only because the cached packets still lack causal per-layer recovery curves.

| Model | Activation top10 drift CV | Drift range | Max global allocation ratio | No-gap fraction | Positive M11b-vs-static traces | Gate |
|---|---:|---:|---:|---:|---:|---|
| Granite | 0.095 | 0.252 | 9.695 | 0.333 | 5/12 | fail |
| Nemotron | 0.119 | 0.249 | 6.245 | 0.167 | 7/12 | pass |
| DeepSeek | 0.180 | 0.350 | 7.266 | 0.083 | 6/12 | pass |
| Falcon | 0.340 | 0.527 | 7.194 | 0.000 | 8/12 | pass |

Dominant final-position layers under the raw global top-10% allocation proxy:

| Model | Dominant layers |
|---|---|
| Granite | 39: 9.695x, 38: 9.007x, 37: 7.305x, 36: 5.202x, 35: 3.880x |
| Nemotron | 51: 6.245x, 50: 5.874x, 49: 5.219x, 48: 4.691x, 47: 4.067x |
| DeepSeek | 27: 7.266x, 26: 5.818x, 25: 4.169x, 24: 3.091x, 23: 2.266x |
| Falcon | 35: 7.194x, 34: 6.466x, 33: 5.107x, 32: 4.136x, 31: 3.097x |

## Membership Churn By Model

Top-10 M11b protected-set churn uses Jaccard distance over cached protected-channel sets. Late adjacent churn excludes the transient start and uses intervals beginning at decode position 3000.

| Model | Layers | First-final drift mean | First-final drift max | Late churn median | Late churn p90 | Most drifting top10 layers |
|---|---:|---:|---:|---:|---:|---|
| Granite | 40 | 0.676 | 0.832 | 0.238 | 0.364 | 0 (0.832), 37 (0.827), 38 (0.806), 39 (0.792), 36 (0.791) |
| Nemotron | 52 | 0.579 | 0.688 | 0.044 | 0.065 | 6 (0.688), 47 (0.681), 3 (0.678), 49 (0.678), 48 (0.675) |
| DeepSeek | 28 | 0.431 | 0.554 | 0.289 | 0.349 | 26 (0.554), 27 (0.533), 21 (0.519), 24 (0.512), 19 (0.505) |
| Falcon | 36 | 0.446 | 0.713 | 0.298 | 0.463 | 35 (0.713), 31 (0.688), 33 (0.688), 32 (0.679), 34 (0.679) |

## No-Gap And Positive-Gap Traces

`positive_method_gap` means `m11b_top10_recovery - static_top10_recovery > 0` on a trace with recoverable static-1% gap.

### DeepSeek

- No-gap fraction: 1/12 = 0.083.
- Positive M11b-vs-static traces: 0, 4, 5, 9, 10, 11.

| Trace | Static gap | M11b top10 | Static top10 | M11b-static | No gap | Drift | EOS |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.003053 | -0.701 | -1.142 | 0.442 | no | 0.794 | none |
| 1 | -0.000647 | n/a | n/a | n/a | yes | 0.797 | none |
| 2 | 0.016102 | 0.510 | 0.729 | -0.219 | no | 0.800 | none |
| 3 | 0.074391 | 0.437 | 0.509 | -0.072 | no | 0.785 | none |
| 4 | 0.000326 | -1.593 | -1.793 | 0.201 | no | 0.787 | none |
| 5 | 0.002091 | 0.721 | 0.487 | 0.233 | no | 0.809 | 3708 |
| 6 | 0.001328 | -0.409 | 0.836 | -1.245 | no | 0.804 | none |
| 7 | 0.004286 | 0.494 | 0.538 | -0.045 | no | 0.794 | none |
| 8 | 0.096555 | 0.335 | 0.377 | -0.041 | no | 0.813 | none |
| 9 | 0.005638 | -0.195 | -0.653 | 0.458 | no | 0.787 | none |
| 10 | 0.000044 | 0.425 | -1.469 | 1.894 | no | 0.745 | none |
| 11 | 0.005312 | -0.092 | -0.102 | 0.009 | no | 0.825 | none |

### Falcon

- No-gap fraction: 0/12 = 0.000.
- Positive M11b-vs-static traces: 0, 2, 4, 6, 7, 8, 9, 11.

| Trace | Static gap | M11b top10 | Static top10 | M11b-static | No gap | Drift | EOS |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.026255 | 0.453 | 0.405 | 0.048 | no | 0.813 | 834 |
| 1 | 0.209011 | -0.143 | -0.033 | -0.110 | no | 0.797 | 9900 |
| 2 | 0.207851 | 0.055 | -0.160 | 0.215 | no | 0.832 | 714 |
| 3 | 0.105688 | -0.065 | -0.017 | -0.048 | no | 0.793 | 2057 |
| 4 | 0.047134 | 0.111 | -0.091 | 0.203 | no | 0.798 | 1027 |
| 5 | 0.084486 | -0.145 | -0.084 | -0.061 | no | 0.760 | 1212 |
| 6 | 0.131456 | 0.242 | 0.168 | 0.073 | no | 0.824 | 910 |
| 7 | 0.030508 | 0.383 | 0.326 | 0.057 | no | 0.842 | 1570 |
| 8 | 0.060069 | -0.599 | -0.791 | 0.193 | no | 0.794 | 1904 |
| 9 | 0.196809 | -0.238 | -0.411 | 0.173 | no | 0.808 | 842 |
| 10 | 0.201525 | 0.165 | 0.172 | -0.007 | no | 0.793 | 711 |
| 11 | 0.122822 | 0.032 | 0.031 | 0.002 | no | 0.792 | 2865 |

## Recommended Smoke Trace Set

Use the following deterministic stratified traces; do not replace them with random draws.

### deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

- Prompt indices: `[5, 11, 8]`
- Prompt IDs: `['opencompass_AIME2025_I_5', 'opencompass_AIME2025_I_11', 'opencompass_AIME2025_I_8']`

| Role | Trace | Static gap | M11b-static | Drift | EOS |
|---|---:|---:|---:|---:|---:|
| positive_gap_anchor | 5 | 0.002091 | 0.233 | 0.809 | 3708 |
| long_high_drift_stress | 11 | 0.005312 | 0.009 | 0.825 | none |
| representative_median | 8 | 0.096555 | -0.041 | 0.813 | none |

### tiiuae/Falcon-H1-0.5B-Instruct

- Prompt indices: `[7, 1, 11]`
- Prompt IDs: `['opencompass_AIME2025_I_7', 'opencompass_AIME2025_I_1', 'opencompass_AIME2025_I_11']`

| Role | Trace | Static gap | M11b-static | Drift | EOS |
|---|---:|---:|---:|---:|---:|
| positive_gap_high_drift_anchor | 7 | 0.030508 | 0.057 | 0.842 | 1570 |
| long_stress | 1 | 0.209011 | -0.110 | 0.797 | 9900 |
| representative_median | 11 | 0.122822 | 0.002 | 0.792 | 2865 |

## Saturated / Alive

- Saturated: Nemotron HYST is low value because top10 late adjacent churn is below the persistent-churn threshold; killed branches remain untouched.
- Alive: DeepSeek/Falcon HYST and LAMBDA smoke gates pass as bounded smoke work only.
- Highest-priority next branch: run only the preregistered HYST/LAMBDA smoke packet on the fixed DeepSeek/Falcon trace set, with matched random controls and no random trace substitution.

## Ruling Updates

- Promoted: DeepSeek/Falcon fixed smoke trace set for HYST/LAMBDA gate testing.
- Weakened: architecture-agnostic M11b positive-method story remains weak because DeepSeek/Falcon base M11b packets are ambiguous, not passes.
- Still blocked: LAMBDA has no cached causal per-layer recovery curve; a smoke win is required before widening.
