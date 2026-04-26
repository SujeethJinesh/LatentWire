# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-26`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp32_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.533142`
- mean best-control margin: `-2.236354`
- mean matched-minus-control margin: `-1.296788`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -12.430716 | -12.341139 | -0.089577 | zero_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.059996 | -10.093343 | -0.966652 | zero_source | control_or_negative |
| 2de1549556000830 | 39 | 33 | -9.673961 | -9.063781 | -0.610180 | zero_source | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.598855 | -2.458839 | -5.140017 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.427335 | 14.946424 | -0.519089 | shuffled_source | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.137342 | 5.592556 | -0.455214 | slots_only | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -6.007460 | -5.724069 | -0.283392 | zero_source | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -11.469300 | -10.719103 | -0.750197 | zero_source | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -3.147762 | -2.374022 | -0.773740 | shuffled_source | control_or_negative |
