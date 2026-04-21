# Process Repair Route Summary

| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected_route_no_repair | 0.1286 | +0.0714 | 0.1286 | 5 | 0 | 4 | 61 | 0.0000 | 0.0000 | 0.0000 | 0.6714 | 0.1571 |
| target_self_repair | 0.1714 | +0.1143 | 0.0571 | 8 | 0 | 4 | 58 | 0.5714 | 0.1143 | 0.0000 | 1.0000 | 0.1571 |
| process_repair_selected_route | 0.2000 | +0.1429 | 0.1286 | 10 | 0 | 4 | 56 | 0.4000 | 0.0714 | 0.0000 | 0.6714 | 0.1571 |

Interpretation:

This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Optional controls log selected-route no-repair and target self-repair under the same repair prompt, so gains can be attributed to candidate generation, target-side repair, or their combination.
