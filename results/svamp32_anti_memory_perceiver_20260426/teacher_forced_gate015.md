# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-25`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_anti_memory_perceiver_20260426/checkpoints/qwen25_to_qwen3_svamp32_anti_memory_w080_ctrl050_am050_r16_b16_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.780554`
- mean best-control margin: `-2.888414`
- mean matched-minus-control margin: `-0.892140`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -12.975049 | -12.835405 | -0.139644 | shuffled_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.578038 | -11.370095 | -0.207943 | slots_only | control_or_negative |
| 2de1549556000830 | 39 | 33 | -10.040082 | -9.789280 | -0.250801 | shuffled_source | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.992219 | -3.560501 | -4.431718 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.346348 | 14.515576 | -0.169229 | zero_source | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.555715 | 5.709219 | -0.153504 | slots_only | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -6.162285 | -5.989544 | -0.172741 | target_only | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -10.573222 | -10.591224 | 0.018001 | zero_source | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -2.491578 | -2.690937 | 0.199360 | shuffled_source | control_or_negative |
