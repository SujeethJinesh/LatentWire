# Process Gate Feature Audit

This audit derives non-oracle process features from the selected-route
solution text, then applies the same high-score-means-skip-repair gate
used by the cheaper metadata audit.

## qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl

| Feature | N | Selected AUROC | Help AUROC | Selected mean | Wrong mean | Help mean | Best threshold | Best acc | Saved repair | Missed help |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| format_plus_process_score | 70 | 0.8707 | 0.4769 | 11.3889 | 6.2186 | 6.1000 | 7.8333 | 0.2000 | 0.3286 | 0 |
| answer_marker_score | 70 | 0.7842 | 0.4077 | 0.6667 | 0.0984 | 0.0000 | 1.0000 | 0.2000 | 0.1714 | 0 |
| process_completeness_score | 70 | 0.6703 | 0.5554 | 6.4444 | 4.8907 | 5.4000 | 7.6667 | 0.2000 | 0.1286 | 0 |
| equation_count | 70 | 0.7787 | 0.5831 | 1.2222 | 0.3770 | 0.6000 | 2.0000 | 0.2000 | 0.1000 | 0 |
| valid_equation_count | 70 | 0.6703 | 0.6231 | 0.8889 | 0.3279 | 0.6000 | 2.0000 | 0.2000 | 0.0714 | 0 |
| reasoning_step_count | 70 | 0.2769 | 0.4831 | 2.6667 | 3.9508 | 3.6000 | 7.0000 | 0.2000 | 0.0714 | 0 |
| equation_valid_fraction | 70 | 0.6302 | 0.6462 | 0.5556 | 0.2951 | 0.6000 | - | - | - | - |
| prediction_tail_match_score | 70 | 0.5082 | 0.5077 | 1.0000 | 0.9836 | 1.0000 | - | - | - | - |
| finished_tail_score | 70 | 0.4936 | 0.5538 | 0.8889 | 0.9016 | 1.0000 | - | - | - | - |

## qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl

| Feature | N | Selected AUROC | Help AUROC | Selected mean | Wrong mean | Help mean | Best threshold | Best acc | Saved repair | Missed help |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| process_completeness_score | 70 | 0.8307 | 0.3232 | 6.5733 | 4.5185 | 4.2821 | 6.3333 | 0.5429 | 0.2286 | 0 |
| format_plus_process_score | 70 | 0.9187 | 0.3090 | 9.0333 | 4.3963 | 4.1667 | 8.5000 | 0.5429 | 0.2143 | 0 |
| answer_marker_score | 70 | 0.6800 | 0.4211 | 0.3600 | 0.0000 | 0.0000 | 1.0000 | 0.5429 | 0.1286 | 0 |
| reasoning_step_count | 70 | 0.6338 | 0.5682 | 6.5600 | 5.9333 | 6.3846 | 9.0000 | 0.5429 | 0.0143 | 0 |
| equation_valid_fraction | 70 | 0.7778 | 0.3981 | 0.6000 | 0.0444 | 0.0769 | - | - | - | - |
| valid_equation_count | 70 | 0.7778 | 0.3981 | 0.6000 | 0.0444 | 0.0769 | - | - | - | - |
| equation_count | 70 | 0.7444 | 0.3718 | 0.6000 | 0.1111 | 0.0769 | - | - | - | - |
| finished_tail_score | 70 | 0.5400 | 0.3691 | 0.8800 | 0.8000 | 0.6154 | - | - | - | - |
| prediction_tail_match_score | 70 | 0.5000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | - | - | - | - |
