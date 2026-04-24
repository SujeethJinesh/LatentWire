# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-24`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.834308`
- mean best-control margin: `-2.637356`
- mean matched-minus-control margin: `-1.196952`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -13.573610 | -12.979069 | -0.594540 | shuffled_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.451377 | -11.165769 | -0.285608 | zero_source | control_or_negative |
| 2de1549556000830 | 39 | 33 | -9.640119 | -9.043700 | -0.596419 | zero_source | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.988647 | -2.830618 | -5.158029 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.211058 | 14.415606 | -0.204548 | slots_only | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.436848 | 5.779413 | -0.342566 | target_only | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -6.034603 | -5.291545 | -0.743058 | zero_source | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -11.195454 | -10.621781 | -0.573673 | shuffled_source | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -3.699000 | -2.794599 | -0.904401 | zero_source | control_or_negative |
