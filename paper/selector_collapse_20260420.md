# Selector Collapse Telemetry (2026-04-20)

Telemetry source: `selector_trace` selected-position overlap from controlled `gsm8k_eval_10` runs. Higher collapse means layers repeatedly transmit the same prompt positions, which is a route-interface warning signal.

| Method | Acc | Route collapse | Layer Jaccard | Unique position frac | Prefix frac | Suffix frac | Full trace frac | Mean score entropy | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynalign_prefdist_attention_stratified | 0.1000 | 1.4469 | 0.5167 | 0.0698 | 0.3612 | 0.3292 | 1.0000 | 1.7304 | four-bin attention-stratified selector coverage ablation |
| dynalign_ctxonly_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | prediction-overlap null for dynalign route selection |
| dynalign_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | best base dynalign interface on controlled gsm8k_eval_10 |
| dynalign_prefdist_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | least-destructive stronger teacher on controlled gsm8k_eval_10 |
| grouped_rotational_transport | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | geometry-side branch that survives the controlled slice |
| module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | direct-output slotted module without token remapping |
| readout_adapter | 0.0000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | prompt-local attention-readout teacher negative boundary |

Current interpretation: the older bridge teachers share the same truncated selector pattern, while the attention-stratified ablation broadens prompt-region coverage but still ties target-alone on controlled GSM10 and loses the GSM5 smoke. Naive coverage balancing is not enough; the next route ablation needs target-query-conditioned slots or head-wise atoms, not only broader top-k position selection.
