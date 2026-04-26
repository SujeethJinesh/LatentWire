# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-25`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_anti_memory_perceiver_20260426/checkpoints/qwen25_to_qwen3_svamp32_anti_memory_w080_ctrl050_am050_r16_b16_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.802045`
- mean best-control margin: `-2.936073`
- mean matched-minus-control margin: `-0.865972`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -13.026940 | -12.994759 | -0.032181 | shuffled_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.512660 | -11.308105 | -0.204554 | slots_only | control_or_negative |
| 2de1549556000830 | 39 | 33 | -10.143353 | -9.811285 | -0.332068 | shuffled_source | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.834768 | -3.613736 | -4.221032 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.309620 | 14.518890 | -0.209270 | zero_source | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.395828 | 5.592556 | -0.196728 | slots_only | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -6.312366 | -6.108584 | -0.203781 | target_only | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -10.454502 | -10.500382 | 0.045879 | zero_source | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -2.379836 | -2.631366 | 0.251530 | shuffled_source | control_or_negative |
