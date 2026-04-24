# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-24`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.764944`
- mean best-control margin: `-2.272635`
- mean matched-minus-control margin: `-1.492309`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -13.347810 | -12.956595 | -0.391215 | zero_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -10.996808 | -11.238231 | 0.241423 | zero_source | control_or_negative |
| 2de1549556000830 | 39 | 33 | -9.659750 | -8.681938 | -0.977812 | zero_source | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -7.854908 | -1.341321 | -6.513587 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.129086 | 14.568278 | -0.439192 | zero_source | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.140525 | 6.013994 | -0.873469 | zero_source | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -5.167335 | -5.577995 | 0.410661 | shuffled_source | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -10.419864 | -10.148226 | -0.271638 | shuffled_source | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -2.471706 | -2.605896 | 0.134190 | zero_source | control_or_negative |
