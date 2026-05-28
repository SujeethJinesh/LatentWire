# Falcon HYST Threshold Analysis

Date: 2026-05-28T06:29:45Z

## Status

- Paper readiness: not ICLR-ready. HYST is a smoke-only positive-method candidate, not evidence.
- Current story: Falcon-H1 has weak M11b endpoint recovery but persistent protected-channel churn, so hysteresis is worth testing only if it can reduce boundary churn without freezing stale channels.
- Blocking gap: no Falcon HYST endpoint scoring exists; this artifact is a CPU-only gate from cached trajectories and activation summaries.

## Inputs

No GPU jobs were run. I read the current ledger/decisions/reviewer feedback, the HYST prereg/runner, the funnel prefilter packet, and the Falcon M11b cache.

Primary Falcon sources:

| Source | Path |
|---|---|
| M11b protected trajectories | `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/08_m11b_falcon/source_files/protected_trajectories.json` |
| Cached activation magnitudes | `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z/activation_magnitudes.jsonl.gz` |
| M11b endpoint metrics | `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/08_m11b_falcon/source_files/metrics.json` |
| Smoke traces | `artifacts/funnel_prefilters/smoke_traces.json` |

Smoke traces remain fixed at prompt indices `[7, 1, 11]`.

## Churn Data

Membership churn is reported two ways:

- slot churn: `|P_t triangle P_prev| / (2 * |P|)`, the protected-slot replacement fraction.
- Jaccard churn: `|P_t triangle P_prev| / |P_t union P_prev|`, matching `docs/hyst_churn.md`.

Available Falcon M11b top-10 data:

| Surface | Window | Slot churn mean | Slot churn median | Slot churn p90 | Jaccard median | Jaccard p90 |
|---|---|---:|---:|---:|---:|---:|
| recorded trajectory, 1000-token snapshots | all | 0.190 | 0.175 | 0.301 | 0.298 | 0.463 |
| recorded trajectory, 1000-token snapshots | late | 0.180 | 0.175 | 0.301 | 0.298 | 0.463 |
| reconstructed M11b rank proxy, 100-token cadence | all | 0.123 | 0.117 | 0.204 | 0.209 | 0.339 |
| reconstructed M11b rank proxy, 100-token cadence | late | 0.124 | 0.117 | 0.204 | 0.209 | 0.339 |

Endpoint baseline remains weak: Falcon M11b top-10 median recovery is `0.04394` with CI95 `[-0.14405, 0.20340]`. This is why HYST is only a churn-risk smoke candidate, not a promoted method.

## Local Pool Stability

I interpret margin `m` as percentage points beyond the top-10 enter threshold:

| Margin | Enter | Exit | Late pool stability median | Late exit retainability median | Late stale median | Late stale p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | top 10% | top 12% | 1.000 | 1.000 | 0.058 | 0.087 |
| 5 | top 10% | top 15% | 1.000 | 1.000 | 0.117 | 0.146 |
| 10 | top 10% | top 20% | 1.000 | 1.000 | 0.165 | 0.204 |

The local high-score pool is stable enough for hysteresis at m=5 and m=10. m=2 is less sticky, but its p10 exit retainability is only `0.167`, so it misses a meaningful tail of would-be exiting channels.

## Margin Comparison

Predicted churn reduction uses the reconstructed 100-token M11b rank proxy. Baseline late slot churn mean is `0.1242`.

| Margin | HYST late slot churn mean | Predicted reduction | Readout |
|---:|---:|---:|---|
| 2 | 0.0609 | 0.510 | conservative but leaves boundary churn |
| 5 | 0.0254 | 0.796 | selected balance |
| 10 | 0.0105 | 0.916 | too sticky for first Falcon smoke |

Selected margin: **m=5**. It cuts most predicted churn while keeping stale protected slots below the m=10 regime. This is the highest-value Falcon HYST smoke candidate.

## Gate Decision

Decision: **run Falcon HYST smoke with m=5**, if the smoke runner supports the selected threshold.

Rationale: Falcon churn is high enough, and the local pool is stable enough, that HYST can plausibly reduce tail risk. The method should still be tested only on the fixed smoke traces `[7, 1, 11]` with BF16, static-1%, M11b top-10, and random-HYST controls.

## Caveats

- This is CPU-only threshold analysis. It does not measure perplexity recovery for HYST.
- Protected trajectories record 1000-token snapshots; the 100-token margin proxy is reconstructed from cached activation magnitudes to match the update cadence.
- The current funnel smoke preregistration and runner describe top-2k exit behavior, equivalent to m=10 under this interpretation. This report recommends m=5 to avoid stale-channel freezing; do not silently substitute m=10 without recording that risk.
- The reviewer-facing blocker remains evaluation quality: a smoke pass would only authorize partial evaluation, not a paper claim.
