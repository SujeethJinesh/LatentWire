# ThoughtFlow-FP8 RDU No-Retuning Reproduction Check

Historical status: **REPRODUCED on the same measured no-retuning surface; later gates still demote rdu_topk.**

- diagnostic type: `measured_no_retuning_rerun_against_cached_frozen_gate`
- cached label: `cached_promoted_gate`
- measured label: `measured_reproduction_rerun`
- method branch: `rdu_topk`
- scored traces: 74
- keep fraction: 0.20
- continuation tokens: 24

This reruns the frozen sparse-cache probe with the existing `rdu_topk` rule and writes a separate measured artifact. It does not retune policy parameters and does not overwrite the cached first-surface frozen gate.

## Cached vs Measured Decision

| Label | RDU NLL | Best compressed | Margin vs R-KV | Paired vs R-KV | Margin vs ThinKV | Paired vs ThinKV | Promotion |
|---|---:|---|---:|---:|---:|---:|---|
| cached_promoted_gate | 3.779 | rdu_topk | +0.160 | -0.160 [-0.264,-0.050] | +0.121 | -0.121 [-0.211,-0.037] | pass |
| measured_reproduction_rerun | 3.779 | rdu_topk | +0.160 | -0.160 [-0.264,-0.050] | +0.121 | -0.121 [-0.211,-0.037] | pass |

## Measured Policy Table

| Policy | NLL | Delta measured-cached |
|---|---:|---:|
| full_cache | 2.848 | +0.000 |
| rdu_topk | 3.779 | +0.000 |
| thin_kv_like | 3.900 | +0.000 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 3.908 | +0.000 |
| thoughtflow_saliency_recent | 3.920 | +0.000 |
| rkv_like | 3.939 | +0.000 |
| longflow_like | 4.158 | +0.000 |

## Strict Separation

Positive margins mean the measured row is worse than `rdu_topk`.

| Family | Policy | Margin NLL vs RDU |
|---|---|---:|
| stopped ThoughtFlow family | thoughtflow_saliency_recent | +0.141 |
| stopped ThoughtFlow family | tf_sparse_r0.55_p0.05_m0.12_a2 | +0.129 |
| cross-family baseline | rkv_like | +0.160 |
| cross-family baseline | thin_kv_like | +0.121 |
| cross-family baseline | longflow_like | +0.379 |

## Oracle And Headroom

- measured per-trace compressed oracle NLL: 3.634
- measured `rdu_topk` NLL: 3.779
- measured full-cache NLL: 2.848
- `rdu_topk` gap to per-trace compressed oracle: 0.145
- per-trace compressed oracle gap to full cache: 0.786
- `rdu_topk` oracle hit rate: 0.419

## Decision

This is a measured reproduction-style check, not a new tuning surface. The next gate remains a larger or independently seeded frozen slice with the same strict reporting.
