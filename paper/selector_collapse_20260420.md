# Selector Collapse Telemetry (2026-04-20)

Telemetry source: `selector_trace` selected-position overlap from controlled `gsm8k_eval_10` runs. Higher collapse means layers repeatedly transmit the same prompt positions, which is a route-interface warning signal.

| Method | Acc | Route collapse | Layer Jaccard | Unique position frac | Prefix frac | Suffix frac | Mean score entropy | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dynalign_ctxonly_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 1.7304 | prediction-overlap null for dynalign route selection |
| dynalign_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 1.7304 | best base dynalign interface on controlled gsm8k_eval_10 |
| dynalign_prefdist_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 1.7304 | least-destructive stronger teacher on controlled gsm8k_eval_10 |
| grouped_rotational_transport | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 1.7304 | geometry-side branch that survives the controlled slice |
| module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 1.7304 | direct-output slotted module without token remapping |
| readout_adapter | 0.0000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 1.7304 | prompt-local attention-readout teacher negative boundary |

Current interpretation: identical collapse values across these variants mean the active selector/interface is shared across otherwise different bridge teachers. A route-atom or query-pool ablation must change this selector geometry, not only the post-selected repair objective.
