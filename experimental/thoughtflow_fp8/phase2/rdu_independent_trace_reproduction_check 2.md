# ThoughtFlow-FP8 RDU Independent-Trace Reproduction Check

Status: **NOT REPRODUCED on independent saved-trace no-retuning surface; inspect same-family/cross-family decision details.**

- diagnostic type: `measured_no_retuning_independent_trace_slice_against_cached_frozen_gate`
- cached label: `cached_promoted_gate`
- measured label: `measured_independent_chat_svamp96`
- method branch: `rdu_topk`
- scored traces: 89
- keep fraction: 0.20
- max length: 96
- continuation tokens: 24

This check reruns the frozen sparse-cache probe with the same `rdu_topk` rule and no policy retuning. Only the saved trace inputs change.

## Trace Inputs

- `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl`
- `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl`
- `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl`

## Cached vs Measured Decision

| Label | RDU NLL | Best compressed | Margin vs R-KV | Paired vs R-KV | Margin vs ThinKV | Paired vs ThinKV | Promotion |
|---|---:|---|---:|---:|---:|---:|---|
| cached_promoted_gate | 3.779 | rdu_topk | +0.160 | -0.160 [-0.264,-0.050] | +0.121 | -0.121 [-0.211,-0.037] | pass |
| measured_independent_chat_svamp96 | 4.014 | rkv_like | -0.032 | +0.032 [-0.071,+0.137] | +0.030 | -0.030 [-0.152,+0.085] | fail |

## Measured Policy Table

| Policy | NLL | Delta measured-cached |
|---|---:|---:|
| full_cache | 2.931 | +0.083 |
| rkv_like | 3.981 | +0.042 |
| rdu_topk | 4.014 | +0.235 |
| thoughtflow_saliency_recent | 4.020 | +0.100 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 4.023 | +0.116 |
| thin_kv_like | 4.043 | +0.143 |
| longflow_like | 4.433 | +0.275 |

## Strict Separation

Positive margins mean the measured row is worse than `rdu_topk`.

| Family | Policy | Margin NLL vs RDU |
|---|---|---:|
| stopped ThoughtFlow family | thoughtflow_saliency_recent | +0.006 |
| stopped ThoughtFlow family | tf_sparse_r0.55_p0.05_m0.12_a2 | +0.010 |
| cross-family baseline | rkv_like | -0.032 |
| cross-family baseline | thin_kv_like | +0.030 |
| cross-family baseline | longflow_like | +0.420 |

## Oracle And Headroom

- measured per-trace compressed oracle NLL: 3.754
- measured `rdu_topk` NLL: 4.014
- measured full-cache NLL: 2.931
- `rdu_topk` gap to per-trace compressed oracle: 0.260
- per-trace compressed oracle gap to full cache: 0.823
- `rdu_topk` oracle hit rate: 0.348

## Failure Decomposition

Rows report `rdu_topk` minus the named baseline; positive is worse for `rdu_topk`. Buckets are tertiles computed on the measured independent surface.

| Group | n | RDU-RKV NLL | RDU-Thin NLL | RDU-oracle gap | RDU beats RKV | RDU oracle hit |
|---|---:|---:|---:|---:|---:|---:|
| all | 89 | +0.032 | -0.030 | +0.260 | 0.483 | 0.348 |
| high_rdu_density | 31 | +0.213 | +0.131 | +0.446 | 0.258 | 0.226 |
| long_prefix | 31 | +0.213 | +0.131 | +0.446 | 0.258 | 0.226 |
| low_rdu_density | 34 | -0.049 | -0.147 | +0.167 | 0.588 | 0.412 |
| mid_prefix | 24 | -0.086 | -0.071 | +0.151 | 0.625 | 0.417 |
| mid_rdu_density | 24 | -0.086 | -0.071 | +0.151 | 0.625 | 0.417 |
| oracle_longflow_like | 5 | -0.023 | -0.136 | +0.214 | 0.600 | 0.000 |
| oracle_rdu_topk | 31 | -0.418 | -0.372 | +0.000 | 1.000 | 1.000 |
| oracle_rkv_like | 16 | +0.498 | +0.036 | +0.498 | 0.000 | 0.000 |
| oracle_tf_sparse_r0.55_p0.05_m0.12_a2 | 22 | +0.174 | +0.107 | +0.312 | 0.318 | 0.000 |
| oracle_thin_kv_like | 11 | +0.104 | +0.365 | +0.365 | 0.182 | 0.000 |
| oracle_thoughtflow_saliency_recent | 4 | +0.751 | +0.657 | +0.794 | 0.000 | 0.000 |
| short_continuation | 89 | +0.032 | -0.030 | +0.260 | 0.483 | 0.348 |
| short_prefix | 34 | -0.049 | -0.147 | +0.167 | 0.588 | 0.412 |

## Decision

- same-family positive separation: True (min margin +0.006)
- cross-family positive separation: False (min margin -0.032)

This is an independent saved-trace reproduction surface, not a policy-tuning surface.
