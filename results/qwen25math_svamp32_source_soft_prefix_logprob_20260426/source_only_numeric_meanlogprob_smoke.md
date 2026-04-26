# SVAMP32 Source Soft-Prefix Logprob Probe

- date: `2026-04-26`
- status: `source_soft_prefix_logprob_fails_gate`
- reference rows: `32`
- prefix len: `2`
- hidden dim: `16`
- epochs: `1`
- clean IDs scored: `6`
- matched-only clean IDs: `1`
- control-leak clean IDs: `4`
- mean matched-minus-control clean: `-0.771126`

## Clean Rows

| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta | Status |
|---|---:|---:|---:|---|---:|---:|---|
| 3e8a5691f5443495 | 1 | 3 | 1.159245 | label_shuffled | 2.361965 | -1.202721 | control_or_negative |
| 1d50b408c8f5cd2c | 949 | 1 | 3.141821 | target_only_prefix | 5.260187 | -2.118366 | control_or_negative |
| de1bf4d142544e5b | 57 | 2 | 2.645924 | label_shuffled | 3.249905 | -0.603981 | control_or_negative |
| 47464cc0b064f172 | 24 | 2 | 4.618119 | label_shuffled | 4.101266 | 0.516853 | matched_only_positive |
| 6e9745b37ab6fc45 | 61 | 600 | -3.758445 | label_shuffled | -3.410089 | -0.348355 | control_or_negative |
| 575d7e83d84c1e67 | 2 | 24 | 0.110940 | target_only_prefix | 0.981128 | -0.870188 | control_or_negative |

## Controls

`matched`, `zero_source`, `shuffled_source`, `same_norm_noise`, `target_only_prefix`, `slots_only_prefix`, `label_shuffled`, and `projected_soft_prompt` are scored for every clean row.
