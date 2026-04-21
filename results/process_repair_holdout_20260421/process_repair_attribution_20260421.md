# Process Repair Attribution

Deterministic bootstrap intervals use example-level resampling within each source file.

## qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl

| Method | N | Accuracy | Pre-repair | Changed answer | Repair help | Repair harm | Target selected | Full oracle | Method-only | Baseline-only | Both correct | Both wrong | Delta vs target | Delta vs target self-repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target_alone | 70 | 0.0571 [0.0143, 0.1143] | - | - | - | - | - | - | - | - | - | - | - | - |
| selected_route_no_repair | 70 | 0.1286 [0.0571, 0.2143] | 0.1286 [0.0571, 0.2143] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.6714 [0.5571, 0.7857] | 0.1571 [0.0714, 0.2429] | 5 | 0 | 4 | 61 | 0.0714 [0.0143, 0.1429] | -0.0429 [-0.1429, 0.0571] |
| target_self_repair | 70 | 0.1714 [0.0857, 0.2714] | 0.0571 [0.0143, 0.1143] | 0.5714 [0.4429, 0.6857] | 0.1143 [0.0429, 0.1857] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.1571 [0.0714, 0.2429] | 8 | 0 | 4 | 58 | 0.1143 [0.0429, 0.2000] | - |
| process_repair_selected_route | 70 | 0.2000 [0.1143, 0.3000] | 0.1286 [0.0571, 0.2143] | 0.4000 [0.2857, 0.5143] | 0.0714 [0.0143, 0.1286] | 0.0000 [0.0000, 0.0000] | 0.6714 [0.5571, 0.7857] | 0.1571 [0.0714, 0.2429] | 10 | 0 | 4 | 56 | 0.1429 [0.0714, 0.2286] | 0.0286 [-0.0429, 0.1143] |

## qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl

| Method | N | Accuracy | Pre-repair | Changed answer | Repair help | Repair harm | Target selected | Full oracle | Method-only | Baseline-only | Both correct | Both wrong | Delta vs target | Delta vs target self-repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target_alone | 70 | 0.3000 [0.2000, 0.4143] | - | - | - | - | - | - | - | - | - | - | - | - |
| selected_route_no_repair | 70 | 0.3571 [0.2429, 0.4714] | 0.3571 [0.2429, 0.4714] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.8429 [0.7571, 0.9286] | 0.5286 [0.4143, 0.6429] | 4 | 0 | 21 | 45 | 0.0571 [0.0143, 0.1143] | -0.1429 [-0.2571, -0.0429] |
| target_self_repair | 70 | 0.5000 [0.3857, 0.6143] | 0.3000 [0.2000, 0.4143] | 0.4571 [0.3429, 0.5714] | 0.2000 [0.1143, 0.3000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.5286 [0.4143, 0.6429] | 14 | 0 | 21 | 35 | 0.2000 [0.1143, 0.3000] | - |
| process_repair_selected_route | 70 | 0.5429 [0.4286, 0.6571] | 0.3571 [0.2429, 0.4714] | 0.4000 [0.2857, 0.5143] | 0.1857 [0.1000, 0.2857] | 0.0000 [0.0000, 0.0000] | 0.8429 [0.7571, 0.9286] | 0.5286 [0.4143, 0.6429] | 17 | 0 | 21 | 32 | 0.2429 [0.1429, 0.3429] | 0.0429 [0.0000, 0.1000] |
