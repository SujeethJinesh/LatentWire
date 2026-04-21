# Frontier Selector Telemetry

Unified fast telemetry for toy selector rows and future real route-pool rows.

- Rows: 2
- Schema fields: selector_method, patch_corr, quant_error_corr, feature_persistence, protected_ids, bit_allocation, help, harm, missed_help, false_prune, bytes, compute, stability
- Best patch-corr selector: universal_dictionary_persistence_protect
- Best help selector: universal_dictionary_persistence_protect

| Selector | Patch corr | Quant-error corr | Feature persistence | Protected ids | Bit allocation | Help | Harm | Missed help | False prune | Bytes | Compute | Stability |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| quant_error_protect | 0.2712 | 0.8831 | 0.2140 | 1,4,7 | {"high_bits": 8, "low_bits": 2, "protected_count": 3, "source": "count_only"} | 0.1458 | 0.0000 | 0.0860 | 0.0190 | 91.0000 | 681.0000 | 0.7200 |
| universal_dictionary_persistence_protect | 0.6078 | 0.4410 | 0.3303 | 0,2,5 | {"high_bits": 8, "low_bits": 2, "protected_count": 3, "source": "count_only"} | 0.2917 | 0.0000 | 0.0000 | 0.1110 | 166.0000 | 446.4000 | 1.0000 |
