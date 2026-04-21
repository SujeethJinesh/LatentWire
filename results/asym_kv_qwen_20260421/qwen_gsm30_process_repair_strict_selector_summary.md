# Process Repair Route Summary

| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| process_repair_selected_route | 0.2333 | +0.1667 | 0.1667 | 5 | 0 | 2 | 23 | 0.4333 | 0.0667 | 0.0000 | 0.7000 | 0.3000 |

Interpretation:

This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Raw repair text, pre/post answers, changed-answer flags, and oracle availability are logged per example.
