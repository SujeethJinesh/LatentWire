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
- mean matched-minus-control clean: `-0.382854`

## Clean Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta | Status |
|---|---:|---:|---:|---|---:|---:|---|
| 3e8a5691f5443495 | 1 | 3 | 0.749460 | label_shuffled | 0.920738 | -0.171278 | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | 2.643000 | same_norm_noise | 2.911374 | -0.268373 | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | 2.298716 | target_only_prefix | 3.234608 | -0.935892 | control_or_negative |
| 47464cc0b064f172 | 24 | 2 | 2.831358 | target_only_prefix | 3.224067 | -0.392709 | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -4.248019 | slots_only_prefix | -3.791304 | -0.456716 | control_or_negative |
| 575d7e83d84c1e67 | 2 | 24 | -3.854072 | shuffled_source | -3.781914 | -0.072158 | control_or_negative |
