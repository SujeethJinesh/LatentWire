# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-26`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp32_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.561617`
- mean best-control margin: `-2.407339`
- mean matched-minus-control margin: `-1.154278`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -12.408109 | -12.426274 | 0.018165 | zero_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.307450 | -10.654936 | -0.652514 | target_only | control_or_negative |
| 2de1549556000830 | 39 | 33 | -9.689269 | -9.235985 | -0.453285 | zero_source | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.861768 | -2.676808 | -5.184959 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.504462 | 14.819145 | -0.314683 | shuffled_source | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.392429 | 5.730822 | -0.338393 | target_only | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -6.015026 | -5.726507 | -0.288519 | zero_source | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -11.392882 | -10.895236 | -0.497646 | zero_source | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -3.000968 | -2.444183 | -0.556785 | shuffled_source | control_or_negative |
