# Repair Gate Feature Audit

This audit uses only selected-route telemetry fields. The best gate threshold
uses the convention: repair when the feature is below the threshold, skip
repair otherwise. Rows are sorted by repair saved while preserving repair-all
accuracy when such a threshold exists.

## qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl

| Feature | N | Selected-correct AUROC | Help AUROC | Selected mean | Wrong mean | Help mean | Best threshold | Best acc | Saved repair | Missed help |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_format_score | 70 | 0.8570 | 0.3985 | 4.9444 | 1.3279 | 0.7000 | 3.5000 | 0.2000 | 0.2714 | 0 |
| selected_candidate_format_delta_vs_target | 70 | 0.7568 | 0.4108 | 3.6667 | 1.1885 | 0.6000 | 4.0000 | 0.2000 | 0.1857 | 0 |
| candidate_numeric_mention_count | 70 | 0.7195 | 0.5523 | 7.6667 | 6.0000 | 6.6000 | 9.0000 | 0.2000 | 0.1571 | 0 |
| candidate_unique_numeric_mention_count | 70 | 0.7659 | 0.6508 | 4.7778 | 3.5410 | 4.2000 | 6.0000 | 0.2000 | 0.0714 | 0 |
| candidate_numeric_consistency_score | 70 | 0.4499 | 0.4846 | 8.2222 | 8.2951 | 8.2000 | 10.0000 | 0.2000 | 0.0714 | 0 |
| candidate_unique_predictions | 70 | 0.6275 | 0.5062 | 3.2222 | 2.8033 | 2.8000 | - | - | - | - |
| candidate_completion_score | 70 | 0.4754 | 0.4631 | 1.8333 | 1.9754 | 1.9000 | - | - | - | - |
| candidate_answer_agreement | 70 | 0.4153 | 0.4631 | 1.5556 | 1.7869 | 1.8000 | - | - | - | - |
| candidate_vote_margin | 70 | 0.3689 | 0.5908 | 0.5556 | 1.0000 | 1.4000 | - | - | - | - |
| candidate_vote_count | 70 | 0.3616 | 0.5354 | 1.6667 | 2.0656 | 2.2000 | - | - | - | - |

## qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl

| Feature | N | Selected-correct AUROC | Help AUROC | Selected mean | Wrong mean | Help mean | Best threshold | Best acc | Saved repair | Missed help |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected_candidate_format_delta_vs_target | 70 | 0.6649 | 0.4035 | 2.1600 | 0.1333 | 0.0000 | 3.0000 | 0.5429 | 0.1571 | 0 |
| candidate_format_score | 70 | 0.7880 | 0.3812 | 2.4600 | -0.1222 | -0.1154 | 4.5000 | 0.5429 | 0.1429 | 0 |
| candidate_numeric_mention_count | 70 | 0.7787 | 0.3522 | 5.8000 | 4.1556 | 4.0769 | - | - | - | - |
| candidate_completion_score | 70 | 0.6093 | 0.3556 | 1.9800 | 1.6111 | 1.4231 | - | - | - | - |
| candidate_answer_agreement | 70 | 0.5591 | 0.4676 | 2.5600 | 2.3333 | 2.3077 | - | - | - | - |
| candidate_vote_count | 70 | 0.5178 | 0.4582 | 2.6000 | 2.5778 | 2.4615 | - | - | - | - |
| candidate_vote_margin | 70 | 0.5133 | 0.4393 | 1.6400 | 1.7556 | 1.4615 | - | - | - | - |
| candidate_unique_predictions | 70 | 0.4920 | 0.5162 | 2.3200 | 2.3333 | 2.3846 | - | - | - | - |
| candidate_unique_numeric_mention_count | 70 | 0.4822 | 0.3219 | 3.2400 | 3.2889 | 2.8462 | - | - | - | - |
| candidate_numeric_consistency_score | 70 | 0.4533 | 0.5803 | 8.0400 | 8.1333 | 8.2308 | - | - | - | - |
