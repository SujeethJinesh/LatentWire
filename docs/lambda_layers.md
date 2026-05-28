# LAMBDA Layer Heterogeneity Readout

Date: 2026-05-28

## Status

Current paper readiness: not ICLR-ready. The active positive-method story is
still gated on HybridKernel; OutlierMigrate remains a measurement/mechanism
line with failed protection interventions. This readout only decides whether
M-LAMBDA layerwise budget waterfilling is killed for homogeneous layers, kept
as a dominant-layer hypothesis, or marked inconclusive.

Decision: **FLAG_DOMINANT_LAYERS**.

Do not kill LAMBDA for layer homogeneity. The cached packets show strong
late-layer dominance under raw activation-magnitude allocation proxies, even
though existing protected budgets are flat by construction and no cached packet
contains causal per-layer marginal recovery curves. LAMBDA remains a guarded
gate, not a promoted method claim.

## Formulas

Let `a_{i,l,c,t}` be the cached activation magnitude for trace `i`, layer `l`,
channel `c`, decode position `t`, and let
`bar a_{l,c,t} = mean_i a_{i,l,c,t}`.

- Top-k set: `S_l,t(k) = top_k_c bar a_{l,c,t}`.
- Strict set-leaving:
  `SL_l = |S_l,100(k1) \\ S_l,T(k1)| / |S_l,100(k1)|`,
  where `k1 = ceil(0.01 * C_l)` and `T` is the packet scoring endpoint.
- Original migration proxy:
  `M_l = |{c in S_l,100(k1): |rank_l,T(c) - rank_l,100(c)| > 2}| / k1`.
- Protected trajectory turnover for budget `q`:
  `TL_l,q = |P_l,q,first \\ P_l,q,final| / |P_l,q,first|`,
  using cached `protected_trajectories.json`.
- Positive-static-gap recovery:
  `R_i(m) = 1 - (PPL_i(m) - PPL_i(BF16)) / (PPL_i(static_1pct) - PPL_i(BF16))`,
  only when the denominator is positive. Otherwise the trace is a no-gap trace.
- Raw magnitude waterfill proxy:
  `A_l = |{(l,c): bar a_{l,c,T} is in global top K}|`,
  where `K = sum_l ceil(0.10 * C_l)`. The reported dominance ratio is
  `A_l / ceil(0.10 * C_l)`.

The raw magnitude waterfill proxy is scale-sensitive across layers. It is a
dominance diagnostic, not causal evidence that reallocating budget improves
perplexity.

## Data Paths

Primary LAMBDA target packets:

- Granite M11b: `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z`
- Nemotron M11b: `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`

Supplemental cached Phase 9 packets:

- DeepSeek V2 M11b: `experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z`
- Falcon V2 M11b: `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z`

Static / position-switching intervention context:

- Phase 3 static union: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z`
- Phase 4 static union: `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z`
- Phase 9 M2: `experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z`

Context read before this decision:

- Reviewer feedback: `paper/reviewer_feedback.md`
- Experiment ledger: `paper/experiment_ledger_20260421.md`
- OutlierMigrate reviewer pack: `experimental/outlier_migrate/paper/reviewer_pack.md`
- Existing layer readouts:
  `experimental/outlier_migrate/phase3/results/layer_stratified_migration.md`,
  `experimental/outlier_migrate/phase9/post_m18_analysis/per_layer_dissection.md`

No GPU jobs were run. The CPU-only readout streamed cached compressed JSON and
retained only positions `100` and the scoring endpoint.

## Primary Packet Summary

| Packet | Checker decision | No-gap traces | Best relevant recovery | Flat protected counts | Strict set-leaving mean / min / max | Raw global-top10 allocation dominance |
|---|---|---:|---:|---:|---:|---:|
| Granite M11b | `PASS_M11B_BUDGET_MATTERS` | 4/12 = 0.333 | `m11b_top5` median 0.449; `m11b_top10` median 0.241 | top10 410/layer, CV 0.000 | 0.336 / 0.195 / 0.439 | max 9.695x flat, min 0.015x, Gini 0.836 |
| Nemotron M11b | `PASS_M11B_NEMOTRON_REPLICATES` | 2/12 = 0.167 | `m11b_top10` median 0.815 | top10 269/layer, CV 0.000 | 0.377 / 0.185 / 0.519 | max 6.245x flat, min 0.000x, Gini 0.754 |
| DeepSeek V2 M11b | `AMBIGUOUS_V2_M11B_DEEPSEEK` | 1/12 = 0.083 | `m11b_top10` median 0.335; `static_top10` median 0.377 | top10 154/layer, CV 0.000 | 0.382 / 0.125 / 0.500 | max 7.266x flat, min 0.013x, Gini 0.761 |
| Falcon V2 M11b | `AMBIGUOUS_V2_M11B_FALCON` | 0/12 = 0.000 | `m11b_top5` median 0.070; `m11b_top10` median 0.044 | top10 103/layer, CV 0.000 | 0.341 / 0.182 / 0.455 | max 7.194x flat, min 0.019x, Gini 0.762 |

Read: protected counts are homogeneous because all cached M11b/static budgets
are flat per layer. The heterogeneity signal comes from set turnover and,
much more strongly, raw final-position activation magnitude concentration.

## Dominant Layers

Raw global-top10 allocation proxy, ratio versus the flat per-layer top10 count:

| Packet | Dominant layers |
|---|---|
| Granite M11b | 39: 3975 channels = 9.695x, 38: 3693 = 9.007x, 37: 2995 = 7.305x, 36: 2133 = 5.202x, 35: 1591 = 3.880x |
| Nemotron M11b | 51: 1680 channels = 6.245x, 50: 1580 = 5.874x, 49: 1404 = 5.219x, 48: 1262 = 4.691x, 47: 1094 = 4.067x |
| DeepSeek V2 M11b | 27: 1119 channels = 7.266x, 26: 896 = 5.818x, 25: 642 = 4.169x, 24: 476 = 3.091x, 23: 349 = 2.266x |
| Falcon V2 M11b | 35: 741 channels = 7.194x, 34: 666 = 6.466x, 33: 526 = 5.107x, 32: 426 = 4.136x, 31: 321 = 3.117x |

Lowest raw global-top10 allocation is near zero in early layers for every
packet. This is the clearest non-homogeneous layer signal and points to a
late-layer waterfill hypothesis.

Layer total-mass shares show the same late-layer pattern:

- Granite: layers 39, 38, 37, 36, 35 are 5.261x, 4.349x, 3.536x, 2.989x,
  2.704x their uniform layer mass share.
- Nemotron: layers 51, 50, 49, 48, 47 are 3.685x, 3.539x, 3.212x, 2.981x,
  2.793x.
- DeepSeek: layers 27, 26, 25, 24, 23 are 3.286x, 2.925x, 2.618x, 2.335x,
  2.030x.
- Falcon: layers 35, 34, 33, 32, 31 are 2.930x, 2.802x, 2.522x, 2.364x,
  2.186x.

## Drift / Set-Leaving

Strict top1 set-leaving is not flat, but it is not as sharply layer-dominated
as raw magnitude allocation:

| Packet | Activation strict set-leaving | Original migration proxy | M11b top1 trajectory leave | M11b top10 trajectory leave |
|---|---:|---:|---:|---:|
| Granite M11b | mean 0.336, min 0.195, max 0.439, CV 0.186 | mean 0.794, min 0.634, max 0.878 | mean 0.331, min 0.195, max 0.439 | mean 0.514, min 0.402, max 0.712 |
| Nemotron M11b | mean 0.377, min 0.185, max 0.519, CV 0.168 | mean 0.694, min 0.481, max 0.852 | mean 0.373, min 0.185, max 0.519 | mean 0.411, min 0.279, max 0.524 |
| DeepSeek V2 M11b | mean 0.382, min 0.125, max 0.500, CV 0.264 | mean 0.641, min 0.188, max 0.875 | mean 0.364, min 0.125, max 0.500 | mean 0.278, min 0.104, max 0.383 |
| Falcon V2 M11b | mean 0.341, min 0.182, max 0.455, CV 0.284 | mean 0.417, min 0.182, max 0.636 | mean 0.311, min 0.182, max 0.455 | mean 0.300, min 0.107, max 0.553 |

Read: drift exists across many layers. It does not by itself say "put most
budget in late layers"; the late-layer dominance comes from magnitude scale.

## Protected-Set Recurrence

The top10 protected channel IDs recur across many layers, so there is also a
channel-index dominance effect:

| Packet | Unique protected channel IDs | Channels in >=25% layers | Channels in >=50% layers | Max occurrence |
|---|---:|---:|---:|---:|
| Granite M11b | 2327 | 459 | 252 | 40/40 layers |
| Nemotron M11b | 1519 | 323 | 176 | 52/52 layers |
| DeepSeek V2 M11b | 387 | 227 | 146 | 28/28 layers |
| Falcon V2 M11b | 296 | 152 | 96 | 36/36 layers |

This weakens a pure "different layer, different channels" story. The cached
sets show many recurring channel IDs, while the raw magnitude scale says late
layers dominate if channels are globally ranked without layer normalization.

## Per-Trace Recovery Summaries

Primary target packets:

### Granite M11b

| Trace | No gap | Static gap | M11b top10 | M11b top5 | Static top10 |
|---:|---|---:|---:|---:|---:|
| 0 | yes | -0.001189 | n/a | n/a | n/a |
| 1 | no | 0.028518 | -0.061 | 1.369 | -53.378 |
| 2 | no | 0.043099 | 0.241 | 1.001 | -0.567 |
| 3 | yes | -0.000060 | n/a | n/a | n/a |
| 4 | no | 0.000772 | -10.485 | -2.722 | -651.191 |
| 5 | no | 0.050718 | 1.000 | 1.000 | 1.000 |
| 6 | yes | -0.000018 | n/a | n/a | n/a |
| 7 | no | 0.010724 | -0.727 | -2.278 | -0.327 |
| 8 | no | 0.156538 | 0.241 | 0.120 | 0.224 |
| 9 | no | 0.000755 | 0.495 | 0.704 | 0.576 |
| 10 | no | 0.133063 | 0.376 | 0.195 | 0.287 |
| 11 | yes | -0.001192 | n/a | n/a | n/a |

### Nemotron M11b

| Trace | No gap | Static gap | M11b top10 | M11b top5 | Static top10 |
|---:|---|---:|---:|---:|---:|
| 0 | no | 0.000901 | 0.254 | 0.692 | 0.814 |
| 1 | no | 0.000387 | 1.266 | 1.019 | 1.025 |
| 2 | no | 11.522735 | 0.757 | 0.453 | 0.675 |
| 3 | yes | -0.059692 | n/a | n/a | n/a |
| 4 | yes | -0.000028 | n/a | n/a | n/a |
| 5 | no | 3.864825 | 0.906 | 0.898 | 0.856 |
| 6 | no | 0.000908 | -0.173 | 0.461 | 0.176 |
| 7 | no | 0.013024 | 0.872 | 0.760 | 0.483 |
| 8 | no | 2.515964 | 0.890 | 0.269 | 0.317 |
| 9 | no | 0.000303 | 0.386 | -0.025 | -0.265 |
| 10 | no | 2.133968 | 0.955 | 0.423 | 0.696 |
| 11 | no | 0.000612 | 0.062 | 0.356 | 0.514 |

Supplemental M11b packets:

- DeepSeek V2 M11b: 1/12 no-gap traces; `m11b_top10` median recovery 0.335,
  `m11b_top5` median -0.177, `static_top10` median 0.377.
- Falcon V2 M11b: 0/12 no-gap traces; `m11b_top10` median recovery 0.044,
  `m11b_top5` median 0.070, `static_top10` median -0.025.

Static / switching context:

- Phase 3 static union: 10/24 no-gap traces; `union_primary` median recovery
  0.000, `magnitude_average` median 0.061, `static_2pct` median 0.000.
- Phase 4 static union: 9/24 no-gap traces; `union_primary` median recovery
  0.000, `magnitude_average` median 0.000, `static_2pct` median 0.000.
- Phase 9 M2: 4/12 no-gap traces; `m2_position_conditional` median -0.867,
  `random_bin_assignment` median -0.199, `static_3pct` median -0.626.

These traces show substantial recovery noise and no-gap saturation. They do
not provide per-layer causal marginal curves, so they cannot prove that a
waterfilled budget will beat flat top10.

## Decision

LAMBDA is **not killed for homogeneous layers**.

Reasons:

1. Per-layer protected counts are homogeneous only because all existing cached
   M11b/static budgets are flat per layer.
2. Per-layer strict set-leaving is nonzero and moderately heterogeneous across
   all cached packets.
3. Raw final-position activation magnitudes are strongly layer-dominant, with
   late layers receiving 6x-10x the flat top10 count under a global top10
   allocation proxy.

But LAMBDA is **not promoted as evidence of a positive method**.

Blocker:

- Cached packets do not contain layerwise causal marginal recovery curves
  `Delta_l(k) = R_l(k) - R_l(k-1)`. The strongest heterogeneity signal is
  scale-sensitive and could be a layer-norm / hook-scale artifact.

## Next Implication

The next exact LAMBDA gate, if reopened, should be a preregistered marginal
curve packet with:

1. same total protected-channel budget as M11b top10;
2. raw-magnitude waterfill and layer-normalized waterfill as separate arms;
3. random layer-waterfilled matched-budget control;
4. per-layer `k_l`, `Delta_l(k)`, and no-gap accounting;
5. explicit report of whether late-layer dominance survives normalization.

Until that packet exists, use this readout only to flag late layers
`Granite 35-39`, `Nemotron 47-51`, `DeepSeek 23-27`, and `Falcon 31-35` as the
dominant-layer hypothesis for LAMBDA. Do not claim a positive method from this
cached evidence alone.
