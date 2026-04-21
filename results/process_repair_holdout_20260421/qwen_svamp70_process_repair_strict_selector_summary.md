# Process Repair Route Summary

| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| process_repair_selected_route | 0.5429 | +0.2429 | 0.3571 | 17 | 0 | 21 | 32 | 0.4000 | 0.1857 | 0.0000 | 0.8429 | 0.5286 |

Interpretation:

This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Raw repair text, pre/post answers, changed-answer flags, and oracle availability are logged per example.
