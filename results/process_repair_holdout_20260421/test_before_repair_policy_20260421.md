# Test-Before-Repair Policy Analysis

Policies are evaluated on existing held-out process-repair telemetry. Gated
policies skip target-side repair when the selected route passes a non-oracle
test, otherwise they use the already logged selected-route repair output.

`oracle_precheck_analysis_only` is an upper bound for a perfect pre-repair
test and must not be used as a method row.

## qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl

| Policy | Threshold | Accuracy | Acc CI | Repair rate | Saved repair | Extra repair chars | Extra repair toks | Delta vs repair-all | Delta vs target self | Repaired help | Missed help | Saved correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| format_plus_process_gate | 7.8333 | 0.2000 | [0.1143, 0.3000] | 0.6714 | 0.3286 | 676.8857 | 42.1714 | 0.0000 | 0.0286 | 5 | 0 | 7 |
| format_plus_process_gate | 8.1667 | 0.2000 | [0.1143, 0.2857] | 0.6857 | 0.3143 | 689.0714 | 42.3429 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_plus_process_gate | 8.5000 | 0.2000 | [0.1143, 0.3000] | 0.7000 | 0.3000 | 701.6143 | 42.9429 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_gate | 3.5000 | 0.2000 | [0.1143, 0.3000] | 0.7286 | 0.2714 | 737.9714 | 46.2714 | 0.0000 | 0.0286 | 5 | 0 | 7 |
| format_plus_process_gate | 8.8333 | 0.2000 | [0.1143, 0.3000] | 0.7571 | 0.2429 | 759.2286 | 46.3571 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_and_completion_gate | 2.5000 | 0.2000 | [0.1143, 0.3000] | 0.7714 | 0.2286 | 781.6286 | 47.9286 | 0.0000 | 0.0286 | 5 | 0 | 3 |
| format_plus_process_gate | 9.1667 | 0.2000 | [0.1143, 0.3000] | 0.7714 | 0.2286 | 773.6000 | 46.8286 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_gate | 4.0000 | 0.2000 | [0.1143, 0.3000] | 0.8000 | 0.2000 | 804.1571 | 50.2857 | 0.0000 | 0.0286 | 5 | 0 | 5 |
| format_plus_process_gate | 10.1667 | 0.2000 | [0.1143, 0.2857] | 0.8000 | 0.2000 | 808.1714 | 48.4714 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_delta_gate | 4.0000 | 0.2000 | [0.1143, 0.3000] | 0.8143 | 0.1857 | 817.6857 | 50.0286 | 0.0000 | 0.0286 | 5 | 0 | 4 |
| format_gate | 4.5000 | 0.2000 | [0.1143, 0.3000] | 0.8143 | 0.1857 | 813.9571 | 50.4000 | 0.0000 | 0.0286 | 5 | 0 | 5 |
| format_plus_process_gate | 11.1667 | 0.2000 | [0.1143, 0.3000] | 0.8286 | 0.1714 | 835.0714 | 50.2286 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_plus_process_gate | 11.1667 | 0.2000 | [0.1143, 0.3000] | 0.8429 | 0.1571 | 849.1143 | 50.7286 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_delta_gate | 4.5000 | 0.2000 | [0.1000, 0.3000] | 0.8571 | 0.1429 | 855.8857 | 52.2714 | 0.0000 | 0.0286 | 5 | 0 | 2 |
| format_delta_gate | 5.0000 | 0.2000 | [0.1143, 0.3000] | 0.8714 | 0.1286 | 865.6857 | 52.3857 | 0.0000 | 0.0286 | 5 | 0 | 2 |
| oracle_precheck_analysis_only | - | 0.2000 | [0.1000, 0.3000] | 0.8714 | 0.1286 | 879.4714 | 52.3286 | 0.0000 | 0.0286 | 5 | 0 | 9 |

## qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl

| Policy | Threshold | Accuracy | Acc CI | Repair rate | Saved repair | Extra repair chars | Extra repair toks | Delta vs repair-all | Delta vs target self | Repaired help | Missed help | Saved correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_precheck_analysis_only | - | 0.5429 | [0.4286, 0.6571] | 0.6429 | 0.3571 | 555.1286 | 41.1429 | 0.0000 | 0.0429 | 13 | 0 | 25 |
| process_gate | 6.3333 | 0.5429 | [0.4286, 0.6571] | 0.7714 | 0.2286 | 661.1429 | 49.2714 | 0.0000 | 0.0429 | 13 | 0 | 15 |
| format_plus_process_gate | 8.5000 | 0.5429 | [0.4286, 0.6571] | 0.7857 | 0.2143 | 666.7000 | 49.9714 | 0.0000 | 0.0429 | 13 | 0 | 15 |
| process_gate | 6.6667 | 0.5429 | [0.4286, 0.6571] | 0.7857 | 0.2143 | 673.6714 | 50.1857 | 0.0000 | 0.0429 | 13 | 0 | 14 |
| process_gate | 7.0000 | 0.5429 | [0.4143, 0.6571] | 0.8000 | 0.2000 | 683.8143 | 51.0143 | 0.0000 | 0.0429 | 13 | 0 | 13 |
| format_plus_process_gate | 8.8333 | 0.5429 | [0.4286, 0.6571] | 0.8143 | 0.1857 | 689.7857 | 51.8000 | 0.0000 | 0.0429 | 13 | 0 | 13 |
| format_plus_process_gate | 9.1667 | 0.5429 | [0.4286, 0.6571] | 0.8286 | 0.1714 | 702.3143 | 52.7143 | 0.0000 | 0.0429 | 13 | 0 | 12 |
| format_delta_gate | 3.0000 | 0.5429 | [0.4143, 0.6571] | 0.8429 | 0.1571 | 713.9286 | 53.5143 | 0.0000 | 0.0429 | 13 | 0 | 9 |
| format_gate | 4.5000 | 0.5429 | [0.4286, 0.6571] | 0.8571 | 0.1429 | 724.9857 | 54.4286 | 0.0000 | 0.0429 | 13 | 0 | 10 |
| format_plus_process_gate | 9.5000 | 0.5429 | [0.4286, 0.6571] | 0.8571 | 0.1429 | 725.8857 | 54.5429 | 0.0000 | 0.0429 | 13 | 0 | 10 |
| format_delta_gate | 5.0000 | 0.5429 | [0.4286, 0.6571] | 0.8857 | 0.1143 | 749.2000 | 56.2571 | 0.0000 | 0.0429 | 13 | 0 | 8 |
| format_plus_process_gate | 12.5000 | 0.5429 | [0.4286, 0.6571] | 0.9000 | 0.1000 | 760.7000 | 57.1714 | 0.0000 | 0.0429 | 13 | 0 | 7 |
| process_gate | 9.0000 | 0.5429 | [0.4143, 0.6571] | 0.9143 | 0.0857 | 772.8857 | 58.0714 | 0.0000 | 0.0429 | 13 | 0 | 6 |
| format_delta_gate | 7.0000 | 0.5429 | [0.4143, 0.6571] | 0.9429 | 0.0571 | 794.2857 | 59.7286 | 0.0000 | 0.0429 | 13 | 0 | 4 |
| format_gate | 7.5000 | 0.5429 | [0.4286, 0.6571] | 0.9429 | 0.0571 | 794.2857 | 59.7286 | 0.0000 | 0.0429 | 13 | 0 | 4 |
| format_plus_process_gate | 13.5000 | 0.5429 | [0.4286, 0.6571] | 0.9429 | 0.0571 | 795.3714 | 59.8143 | 0.0000 | 0.0429 | 13 | 0 | 4 |
