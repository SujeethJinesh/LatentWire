# Test-Before-Repair Policy Analysis

Policies are evaluated on existing held-out process-repair telemetry. Gated
policies skip target-side repair when the selected route passes a non-oracle
test, otherwise they use the already logged selected-route repair output.

`oracle_precheck_analysis_only` is an upper bound for a perfect pre-repair
test and must not be used as a method row.

## qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl

| Policy | Threshold | Accuracy | Repair rate | Saved repair | Delta vs repair-all | Delta vs target self | Repaired help | Missed help | Saved correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| format_plus_process_gate | 7.8333 | 0.2000 | 0.6714 | 0.3286 | 0.0000 | 0.0286 | 5 | 0 | 7 |
| format_plus_process_gate | 8.1667 | 0.2000 | 0.6857 | 0.3143 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_plus_process_gate | 8.5000 | 0.2000 | 0.7000 | 0.3000 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_gate | 3.5000 | 0.2000 | 0.7286 | 0.2714 | 0.0000 | 0.0286 | 5 | 0 | 7 |
| format_plus_process_gate | 8.8333 | 0.2000 | 0.7571 | 0.2429 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_and_completion_gate | 2.5000 | 0.2000 | 0.7714 | 0.2286 | 0.0000 | 0.0286 | 5 | 0 | 3 |
| format_plus_process_gate | 9.1667 | 0.2000 | 0.7714 | 0.2286 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_gate | 4.0000 | 0.2000 | 0.8000 | 0.2000 | 0.0000 | 0.0286 | 5 | 0 | 5 |
| format_plus_process_gate | 10.1667 | 0.2000 | 0.8000 | 0.2000 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_delta_gate | 4.0000 | 0.2000 | 0.8143 | 0.1857 | 0.0000 | 0.0286 | 5 | 0 | 4 |
| format_gate | 4.5000 | 0.2000 | 0.8143 | 0.1857 | 0.0000 | 0.0286 | 5 | 0 | 5 |
| format_plus_process_gate | 11.1667 | 0.2000 | 0.8286 | 0.1714 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_plus_process_gate | 11.1667 | 0.2000 | 0.8429 | 0.1571 | 0.0000 | 0.0286 | 5 | 0 | 6 |
| format_delta_gate | 4.5000 | 0.2000 | 0.8571 | 0.1429 | 0.0000 | 0.0286 | 5 | 0 | 2 |
| format_delta_gate | 5.0000 | 0.2000 | 0.8714 | 0.1286 | 0.0000 | 0.0286 | 5 | 0 | 2 |
| oracle_precheck_analysis_only | - | 0.2000 | 0.8714 | 0.1286 | 0.0000 | 0.0286 | 5 | 0 | 9 |

## qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl

| Policy | Threshold | Accuracy | Repair rate | Saved repair | Delta vs repair-all | Delta vs target self | Repaired help | Missed help | Saved correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_precheck_analysis_only | - | 0.5429 | 0.6429 | 0.3571 | 0.0000 | 0.0429 | 13 | 0 | 25 |
| process_gate | 6.3333 | 0.5429 | 0.7714 | 0.2286 | 0.0000 | 0.0429 | 13 | 0 | 15 |
| format_plus_process_gate | 8.5000 | 0.5429 | 0.7857 | 0.2143 | 0.0000 | 0.0429 | 13 | 0 | 15 |
| process_gate | 6.6667 | 0.5429 | 0.7857 | 0.2143 | 0.0000 | 0.0429 | 13 | 0 | 14 |
| process_gate | 7.0000 | 0.5429 | 0.8000 | 0.2000 | 0.0000 | 0.0429 | 13 | 0 | 13 |
| format_plus_process_gate | 8.8333 | 0.5429 | 0.8143 | 0.1857 | 0.0000 | 0.0429 | 13 | 0 | 13 |
| format_plus_process_gate | 9.1667 | 0.5429 | 0.8286 | 0.1714 | 0.0000 | 0.0429 | 13 | 0 | 12 |
| format_delta_gate | 3.0000 | 0.5429 | 0.8429 | 0.1571 | 0.0000 | 0.0429 | 13 | 0 | 9 |
| format_gate | 4.5000 | 0.5429 | 0.8571 | 0.1429 | 0.0000 | 0.0429 | 13 | 0 | 10 |
| format_plus_process_gate | 9.5000 | 0.5429 | 0.8571 | 0.1429 | 0.0000 | 0.0429 | 13 | 0 | 10 |
| format_delta_gate | 5.0000 | 0.5429 | 0.8857 | 0.1143 | 0.0000 | 0.0429 | 13 | 0 | 8 |
| format_plus_process_gate | 12.5000 | 0.5429 | 0.9000 | 0.1000 | 0.0000 | 0.0429 | 13 | 0 | 7 |
| process_gate | 9.0000 | 0.5429 | 0.9143 | 0.0857 | 0.0000 | 0.0429 | 13 | 0 | 6 |
| format_delta_gate | 7.0000 | 0.5429 | 0.9429 | 0.0571 | 0.0000 | 0.0429 | 13 | 0 | 4 |
| format_gate | 7.5000 | 0.5429 | 0.9429 | 0.0571 | 0.0000 | 0.0429 | 13 | 0 | 4 |
| format_plus_process_gate | 13.5000 | 0.5429 | 0.9429 | 0.0571 | 0.0000 | 0.0429 | 13 | 0 | 4 |
