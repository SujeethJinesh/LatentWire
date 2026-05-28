# HYST Churn Analysis

Date: 2026-05-28

## Scope

This is a CPU-only diagnostic over cached M11b protected-set trajectories. It
does not run new model inference, does not launch GPU jobs, and does not update
the experiment ledger.

Inputs used:

| Model | Trajectory path |
|---|---|
| Granite-4.0-H-Small | `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/protected_trajectories.json` |
| Nemotron-3-Nano-30B-A3B | `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z/protected_trajectories.json` |
| DeepSeek-R1-Distill-Qwen-1.5B | `experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z/protected_trajectories.json` |
| Falcon-H1 | `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z/protected_trajectories.json` |

Not used: M-PRED extension trajectories and the older Granite M11 trajectory,
because this task asks for cached M11b protected-set churn.

## Formula

For each model, regime, layer, and consecutive recorded snapshot pair:

`churn(t) = |P_t symmetric_difference P_prev| / |P_t union P_prev|`

The trajectory files record snapshots at positions
`100, 1000, 2000, ..., 10000`; `P_prev` is the previous recorded snapshot, not
every raw update at cadence 100. The table reports Jaccard-distance churn across
all layers and consecutive recorded intervals. The late window excludes the
first two transient intervals and uses intervals `3000->4000` through
`9000->10000`.

For fixed-size protected sets, a Jaccard churn of `c` corresponds to an
entering/leaving fraction of approximately `c / (2 - c)` on each side. For
example, churn `0.30` means about `17.6%` of the protected set changes.

## Numeric Churn Summary

| Model | Regime | All n | All mean | All median | All p90 | Late n | Late mean | Late median | Late p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Granite | top1 | 400 | 0.185 | 0.178 | 0.327 | 280 | 0.143 | 0.136 | 0.217 |
| Granite | top5 | 400 | 0.201 | 0.178 | 0.360 | 280 | 0.163 | 0.145 | 0.285 |
| Granite | top10 | 400 | 0.312 | 0.272 | 0.534 | 280 | 0.236 | 0.238 | 0.364 |
| Nemotron | top1 | 520 | 0.099 | 0.071 | 0.312 | 364 | 0.031 | 0.000 | 0.071 |
| Nemotron | top5 | 520 | 0.098 | 0.043 | 0.292 | 364 | 0.035 | 0.029 | 0.058 |
| Nemotron | top10 | 520 | 0.115 | 0.051 | 0.290 | 364 | 0.043 | 0.044 | 0.065 |
| DeepSeek | top1 | 280 | 0.244 | 0.222 | 0.400 | 196 | 0.224 | 0.222 | 0.316 |
| DeepSeek | top5 | 280 | 0.324 | 0.317 | 0.412 | 196 | 0.297 | 0.289 | 0.362 |
| DeepSeek | top10 | 280 | 0.303 | 0.298 | 0.379 | 196 | 0.287 | 0.289 | 0.349 |
| Falcon | top1 | 360 | 0.303 | 0.308 | 0.429 | 252 | 0.291 | 0.308 | 0.429 |
| Falcon | top5 | 360 | 0.318 | 0.323 | 0.424 | 252 | 0.304 | 0.295 | 0.375 |
| Falcon | top10 | 360 | 0.308 | 0.298 | 0.463 | 252 | 0.296 | 0.298 | 0.463 |

Model-level aggregates over all three M11b regimes:

| Model | All n | All mean | All median | All p90 | Late n | Late mean | Late median | Late p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Granite | 1200 | 0.233 | 0.200 | 0.426 | 840 | 0.181 | 0.168 | 0.292 |
| Nemotron | 1560 | 0.104 | 0.051 | 0.312 | 1092 | 0.037 | 0.029 | 0.071 |
| DeepSeek | 840 | 0.290 | 0.298 | 0.400 | 588 | 0.270 | 0.279 | 0.362 |
| Falcon | 1080 | 0.310 | 0.308 | 0.439 | 756 | 0.297 | 0.308 | 0.429 |

## Local-Pool Stability Read

Nemotron is locally stable after the initial transient. Its late medians are
`0.000`, `0.029`, and `0.044`, with late p90 at or below `0.071`. A hysteresis
method has little membership-stability headroom on Nemotron.

Granite is mixed. Top1/top5 settle to moderate churn, but top10 remains
unstable enough to matter: late median `0.238`, late p90 `0.364`. Hysteresis
could reduce mechanical turnover there, but Granite's existing M11b recovery is
wide/uncertain, so churn reduction alone is not evidence of quality headroom.

DeepSeek and Falcon are not locally stable. Late median churn stays around
`0.28-0.31` for most regimes, with late p90 up to `0.362` on DeepSeek and
`0.463` on Falcon. This is persistent protected-pool turnover rather than only
startup churn.

## Decision

Decision: **KEEP/INCONCLUSIVE; do not kill HYST globally.**

Kill criterion used for this diagnostic: kill HYST only if late-window churn is
already low across the available M11b packets, operationalized as late median
churn `<= 0.10` and late p90 `<= 0.20` for all model/budget regimes. That
condition is false. Nemotron is saturated, but Granite top10, DeepSeek, and
Falcon still show substantial late protected-set churn.

## Implication

HYST is not promoted as a positive method by this analysis. The cached
trajectories only show that membership instability remains a plausible failure
surface on DeepSeek/Falcon and partly on Granite top10. If HYST is reopened, the
next gate should be a preregistered matched-budget quality replay on the high
churn surfaces, with static/M11b controls and no post-hoc threshold tuning.
Running it on Nemotron first would be low value because the local protected pool
has already stabilized there.
