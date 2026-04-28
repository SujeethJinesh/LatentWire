# Source-Private Hidden-Repair Seed-Repeat Summary

- gate: `source_private_hidden_repair_packet_seed_repeat_20260429`
- pass gate: `True`
- surfaces: `4`
- primary rows: `8`
- destruction rows: `4`
- min primary delta-target lower bound: `0.516`
- min primary delta-control lower bound: `0.506`
- max destruction matched accuracy: `0.250`

| Surface | Family set | Seed | Model | Mode | Pass | Matched | Target | Best control | Valid | Delta target 95% CI | Delta control 95% CI |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.808 | 0.250 | 0.252 | 0.776 | [0.516, 0.600] | [0.514, 0.602] |
| core_seed29 | core | 29 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.252 | 1.000 | [0.714, 0.788] | [0.708, 0.786] |
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.252 | 0.000 | [0.000, 0.000] | [-0.006, 0.000] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.808 | 0.250 | 0.256 | 0.776 | [0.516, 0.602] | [0.506, 0.594] |
| core_seed31 | core | 31 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.256 | 1.000 | [0.710, 0.786] | [0.704, 0.780] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.256 | 0.000 | [0.000, 0.000] | [-0.014, 0.000] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.922 | 0.250 | 0.258 | 0.864 | [0.632, 0.712] | [0.622, 0.706] |
| holdout_seed30 | holdout | 30 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.258 | 1.000 | [0.710, 0.788] | [0.702, 0.778] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.258 | 0.000 | [0.000, 0.000] | [-0.016, -0.002] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.924 | 0.250 | 0.252 | 0.860 | [0.634, 0.716] | [0.632, 0.712] |
| holdout_seed32 | holdout | 32 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.252 | 1.000 | [0.710, 0.786] | [0.710, 0.786] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.252 | 0.000 | [0.000, 0.000] | [-0.006, 0.000] |

## By Model

- `Qwen/Qwen3-0.6B`: min matched `0.808`, mean matched `0.866`, min delta-target lower `0.516`, min valid `0.776`
- `microsoft/Phi-3-mini-4k-instruct`: min matched `1.000`, mean matched `1.000`, min delta-target lower `0.710`, min valid `1.000`

Pass rule: All trace_no_hint primary rows pass across core/holdout seeds; all raw_log_no_trace rows fail; minimum paired lower bound over target-only exceeds +0.15.
