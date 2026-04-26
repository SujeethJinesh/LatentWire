# SVAMP32 Source Cross-Attention Logprob Probe

- date: `2026-04-26`
- status: `source_cross_attention_logprob_fails_gate`
- reference rows: `70`
- prefix len: `2`
- hidden dim: `16`
- epochs: `1`
- clean IDs scored: `6`
- matched-only clean IDs: `0`
- control-leak clean IDs: `3`
- mean matched-minus-control clean: `-0.443233`

## Clean Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta | Status |
|---|---:|---:|---:|---|---:|---:|---|
| 14bfbfc94f2c2e7b | 3 | 16 | -3.732854 | label_shuffled | -3.539630 | -0.193223 | control_or_negative |
| 2de1549556000830 | 39 | 33 | -1.381036 | zero_source | -0.883861 | -0.497175 | control_or_negative |
| 4d780f825bb8541c | 26 | 1 | 0.590536 | target_only_prefix | 1.628020 | -1.037484 | control_or_negative |
| 41cce6c6e6bb0058 | 10 | 3 | 2.748554 | label_shuffled | 3.391998 | -0.643444 | control_or_negative |
| ce08a3a269bf0151 | 2 | 9 | 0.807250 | shuffled_source | 0.918560 | -0.111310 | control_or_negative |
| bd9d8da923981d69 | 22 | 317 | -4.559548 | zero_source | -4.382788 | -0.176761 | control_or_negative |
