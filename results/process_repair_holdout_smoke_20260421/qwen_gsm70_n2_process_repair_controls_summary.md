# Process Repair Route Summary

| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected_route_no_repair | 0.5000 | +0.0000 | 0.5000 | 0 | 0 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| target_self_repair | 1.0000 | +0.5000 | 0.5000 | 1 | 0 | 1 | 0 | 0.5000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |
| process_repair_selected_route | 1.0000 | +0.5000 | 0.5000 | 1 | 0 | 1 | 0 | 0.5000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |

Interpretation:

This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Optional controls log selected-route no-repair and target self-repair under the same repair prompt, so gains can be attributed to candidate generation, target-side repair, or their combination.
