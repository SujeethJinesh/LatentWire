# SVAMP32 Source Cross-Attention Logprob Probe

- date: `2026-04-26`
- status: `source_cross_attention_logprob_fails_gate`
- reference rows: `32`
- prefix len: `2`
- hidden dim: `16`
- epochs: `1`
- clean IDs scored: `6`
- matched-only clean IDs: `0`
- control-leak clean IDs: `4`
- mean matched-minus-control clean: `-0.383649`

## Clean Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta | Status |
|---|---:|---:|---:|---|---:|---:|---|
| 3e8a5691f5443495 | 1 | 3 | 0.641737 | label_shuffled | 0.906638 | -0.264901 | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | 2.516997 | shuffled_source | 2.940691 | -0.423693 | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | 2.193731 | target_only_prefix | 2.990877 | -0.797146 | control_or_negative |
| 47464cc0b064f172 | 24 | 2 | 2.798158 | target_only_prefix | 3.213963 | -0.415805 | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -4.121853 | target_only_prefix | -3.899642 | -0.222211 | control_or_negative |
| 575d7e83d84c1e67 | 2 | 24 | -4.371127 | label_shuffled | -4.192992 | -0.178135 | control_or_negative |
