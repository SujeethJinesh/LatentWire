# Process Repair Route Summary

| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| process_repair_selected_route | 1.0000 | +0.5000 | 0.5000 | 1 | 0 | 1 | 0 | 0.5000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |

Interpretation:

This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Raw repair text, pre/post answers, changed-answer flags, and oracle availability are logged per example.
