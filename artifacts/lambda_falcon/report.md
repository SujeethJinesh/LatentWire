# Falcon LAMBDA Budget Prior

## Status

- Paper readiness: not ICLR-ready; this is only a CPU-side prior artifact for a possible OutlierMigrate smoke, not positive-method evidence.
- Current story: LAMBDA tests whether the M11b top-10 total protected-channel budget can be reallocated across layers better than the flat 103 channels/layer baseline.
- Blocking gap: Falcon-H1 has no cached causal per-layer recovery curves, and its M11b top-10 baseline remains weak.

## Available Signals

- Fixed Falcon smoke traces: `[7, 1, 11]` from `artifacts/funnel_prefilters/smoke_traces.json`.
- Falcon M11b top-10 baseline: median recovery `0.043940`, mean `0.021028`, CI95 `[-0.144051, 0.203398]`.
- No-gap fraction: `0.000` (`0/12` traces).
- Positive M11b-vs-static traces in the prefilter: `8/12`.
- Phase 7 Falcon migration drift by layer has slope `-0.004213` per layer; lower drift in late layers is the verified stability signal used here.
- Prefilter activation/global allocation signal is much sharper: activation top-10 drift CV `0.340354`, range `0.527500`, max global allocation ratio `7.194175`.

## Formula

I used the cached Phase 7 Falcon layer migration fraction as the prior drift value and preserved the total M11b top-10 budget:

```text
stability_l = 1 - drift_l
budget_l = largest_remainder_round(3708 * stability_l / sum_j stability_j)
```

where `3708 = 36 layers * 103 channels/layer`. This intentionally regularizes away from the prefilter's very aggressive global-waterfill allocation because that allocation is not a causal per-layer recovery estimate.

## Chosen Budgets

- Total budget: `3708` channels; flat baseline total: `3708`.
- Budget range: `68` to `160` channels/layer.
- L1 movement from flat M11b top-10: `700` channels (`0.189` of total budget).
- Highest-budget layers: 34:160, 33:155, 35:151, 32:142, 31:141, 30:130.
- Lowest-budget layers: 7:68, 8:72, 10:77, 6:79, 5:81, 9:81.

Full per-layer rows are in `lambda_layer_budget.csv`.

## Gate Decision

Do **not** recommend a standalone Falcon LAMBDA GPU smoke from this prior alone.

Reason: the reallocation is meaningful as a prior, but Falcon-H1 is not routable, the all-trace M11b top-10 median is only `0.043940`, and cached data still lacks per-layer recovery/headroom showing that moving budget to late stable layers should rescue traces `[7, 1, 11]`. If a coordinator already runs the fixed DeepSeek/Falcon funnel smoke packet, `smoke_config_lambda_falcon.json` is runnable and auditable; otherwise defer Falcon-only GPU spend.

## Caveats

- No huge tensors were rederived; this uses compact summaries and existing prefilter artifacts only.
- The Phase 7 migration-drift signal and the prefilter activation-allocation signal both favor late layers, but they measure different objects and should not be averaged into a causal claim.
- This artifact does not alter `RUN_LEDGER.md`, `DECISIONS.md`, paper files, or shared state.
