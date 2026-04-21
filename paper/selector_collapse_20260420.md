# Selector Collapse Telemetry (2026-04-20)

Telemetry source: `selector_trace` selected-position overlap from controlled `gsm8k_eval_10` runs. Higher collapse means layers repeatedly transmit the same prompt positions, which is a route-interface warning signal.

| Method | Acc | Route collapse | Layer Jaccard | Unique position frac | Prefix frac | Suffix frac | Full trace frac | Mean score entropy | Pool entropy | Pool top weight | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynalign_prefdist_attention_stratified | 0.1000 | 1.4469 | 0.5167 | 0.0698 | 0.3612 | 0.3292 | 1.0000 | 1.7304 | 0.0000 | 0.0000 | four-bin attention-stratified selector coverage ablation |
| dynalign_prefdist_query_pool_transport | 0.1000 | 1.4036 | 0.4737 | 0.0702 | 0.3363 | 0.3290 | 1.0000 | 1.7304 | 0.5425 | 0.7188 | attention-pooled representative-slot interface with fixed cache length |
| dynalign_ctxonly_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | 0.0000 | 0.0000 | prediction-overlap null for dynalign route selection |
| dynalign_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | 0.0000 | 0.0000 | best base dynalign interface on controlled gsm8k_eval_10 |
| dynalign_prefdist_module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | 0.0000 | 0.0000 | least-destructive stronger teacher on controlled gsm8k_eval_10 |
| grouped_rotational_transport | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | 0.0000 | 0.0000 | geometry-side branch that survives the controlled slice |
| module_replace | 0.1000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | 0.0000 | 0.0000 | direct-output slotted module without token remapping |
| readout_adapter | 0.0000 | 1.3724 | 0.4603 | 0.0879 | 0.9080 | 0.0007 | 0.0000 | 1.7304 | 0.0000 | 0.0000 | prompt-local attention-readout teacher negative boundary |

Current interpretation: the older bridge teachers share the same truncated selector pattern, while the attention-stratified and fixed query-pool-style ablations broaden prompt-region coverage but still tie target-alone on controlled GSM10 and lose the GSM5 smoke. Naive coverage balancing and deterministic pooled representative slots are not enough; the next route ablation needs learned target-query-conditioned slots, head-wise atoms, or tokenizer-independent byte probes.
