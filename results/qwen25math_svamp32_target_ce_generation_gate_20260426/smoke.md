# SVAMP32 Source Cross-Attention Logprob Probe

- date: `2026-04-26`
- status: `source_cross_attention_logprob_fails_gate`
- reference rows: `32`
- prefix len: `2`
- hidden dim: `16`
- epochs: `1`
- training objective: `target_ce`
- clean IDs scored: `6`
- matched-only clean IDs: `0`
- control-leak clean IDs: `4`
- mean matched-minus-control clean: `-0.194783`

## Clean Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta | Status |
|---|---:|---:|---:|---|---:|---:|---|
| 3e8a5691f5443495 | 1 | 3 | 0.665291 | slots_only_prefix | 0.877297 | -0.212007 | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | 2.720225 | same_norm_noise | 2.897103 | -0.176878 | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | 2.046978 | target_only_prefix | 2.766883 | -0.719905 | control_or_negative |
| 47464cc0b064f172 | 24 | 2 | 2.620258 | target_only_prefix | 3.200406 | -0.580148 | control_or_negative |
| 6e9745b37ab6fc45 | 61 | 600 | -3.426079 | shuffled_source | -4.059072 | 0.632993 | control_or_negative |
| 575d7e83d84c1e67 | 2 | 24 | -4.256545 | label_shuffled | -4.143793 | -0.112752 | control_or_negative |

## Generation Gate

- matched-only clean IDs: `0`
- control-leak clean IDs: `1`

| Condition | Correct | Clean Correct | Target-Self Correct | Numeric Coverage | Empty |
|---|---:|---:|---:|---:|---:|
| matched | 1 | 1 | 0 | 6 | 0 |
| zero_source | 2 | 2 | 0 | 6 | 0 |
| shuffled_source | 2 | 2 | 0 | 6 | 0 |
| target_only_prefix | 2 | 2 | 0 | 6 | 0 |
| slots_only_prefix | 2 | 2 | 0 | 6 | 0 |
