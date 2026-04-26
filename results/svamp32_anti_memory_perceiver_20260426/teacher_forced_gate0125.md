# SVAMP32 Teacher-Forced Connector Diagnostic

- date: `2026-04-25`
- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_anti_memory_perceiver_20260426/checkpoints/qwen25_to_qwen3_svamp32_anti_memory_w080_ctrl050_am050_r16_b16_seed1.pt`
- clean IDs scored: `6`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.766860`
- mean best-control margin: `-2.877031`
- mean matched-minus-control margin: `-0.889829`

## Clean Residual Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 13cb77b698eeadb5 | 8142 | 46 | -12.976549 | -12.836021 | -0.140528 | shuffled_source | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | -11.593971 | -11.406056 | -0.187915 | slots_only | control_or_negative |
| 2de1549556000830 | 39 | 33 | -9.965590 | -9.747859 | -0.217731 | slots_only | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -8.059461 | -3.549498 | -4.509962 | shuffled_source | control_or_negative |
| aee922049c757331 | 1 | 17 | 14.363925 | 14.514388 | -0.150463 | zero_source | control_or_negative |
| e3ab8666238a289e | 1 | 4 | 5.630486 | 5.762863 | -0.132377 | slots_only | control_or_negative |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 4c84ebf42812703b | 10 | 2 | -6.106539 | -5.957247 | -0.149292 | target_only | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | -10.630623 | -10.647663 | 0.017041 | zero_source | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | -2.572726 | -2.749577 | 0.176850 | shuffled_source | control_or_negative |
