# SVAMP70 Teacher-Forced Connector Diagnostic

- date: `2026-04-26`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp70_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
- clean IDs scored: `10`
- matched-positive clean IDs: `4`
- matched-only clean IDs: `0`
- control-leak clean IDs: `4`
- mean matched margin: `-3.404953`
- mean best-control margin: `-2.689198`
- mean matched-minus-control margin: `-0.715756`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -13.295330 | -13.047574 | -0.247756 | shuffled_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.185175 | -11.139697 | -0.045479 | shuffled_source | control_or_negative |
| 2de1549556000830 | 39 | 33 | -9.658085 | -9.409326 | -0.248759 | target_only | control_or_negative |
| 3c5aeb08941dbb6d | 139 | 13 | -0.681989 | -0.679417 | -0.002572 | zero_source | control_or_negative |
| 575d7e83d84c1e67 | 2 | 24 | 2.156248 | 2.605409 | -0.449162 | slots_only | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.975121 | -3.878347 | -4.096773 | shuffled_source | control_or_negative |
| 9325c4efa96bdbca | 2 | 6 | 1.380669 | 1.560032 | -0.179363 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.730146 | 14.822264 | -0.092117 | target_only | control_or_negative |
| dcf26d3b6ad06c6c | 366 | 408 | -15.062374 | -13.453888 | -1.608486 | target_only | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.541479 | 5.728569 | -0.187090 | target_only | control_or_negative |
