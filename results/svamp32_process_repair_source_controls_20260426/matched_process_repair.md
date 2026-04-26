# Process Repair Route Summary

| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected_route_no_repair | 0.2500 | +0.0000 | 0.2500 | 0 | 0 | 8 | 24 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.3125 |
| process_repair_selected_route | 0.3125 | +0.0625 | 0.2500 | 2 | 0 | 8 | 22 | 0.4375 | 0.0625 | 0.0000 | 1.0000 | 0.3125 |

Interpretation:

This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Optional controls log selected-route no-repair and target self-repair under the same repair prompt, so gains can be attributed to candidate generation, target-side repair, or their combination.
