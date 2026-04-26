# Process Repair Route Summary

| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected_route_no_repair | 0.3143 | +0.0143 | 0.3143 | 1 | 0 | 21 | 48 | 0.0000 | 0.0000 | 0.0000 | 0.8857 | 0.4571 |
| target_self_repair | 0.5000 | +0.2000 | 0.3000 | 14 | 0 | 21 | 35 | 0.4571 | 0.2000 | 0.0000 | 1.0000 | 0.4571 |
| process_repair_selected_route | 0.5000 | +0.2000 | 0.3143 | 14 | 0 | 21 | 35 | 0.4143 | 0.1857 | 0.0000 | 0.8857 | 0.4571 |

Interpretation:

This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Optional controls log selected-route no-repair and target self-repair under the same repair prompt, so gains can be attributed to candidate generation, target-side repair, or their combination.
