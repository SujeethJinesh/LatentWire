# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-26`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp32_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.576508`
- mean best-control margin: `-2.475364`
- mean matched-minus-control margin: `-1.101144`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -12.469838 | -12.508939 | 0.039101 | zero_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.393546 | -10.796926 | -0.596620 | target_only | control_or_negative |
| 2de1549556000830 | 39 | 33 | -9.673416 | -9.300223 | -0.373194 | zero_source | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.962468 | -2.815976 | -5.146492 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.527364 | 14.762724 | -0.235360 | shuffled_source | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.512854 | 5.807156 | -0.294302 | target_only | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -6.013057 | -5.764067 | -0.248990 | zero_source | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -11.324464 | -10.935296 | -0.389168 | slots_only | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -2.969334 | -2.517924 | -0.451410 | shuffled_source | control_or_negative |
