# ThoughtFlow-FP8 RDU Alternate-Surface Reproduction Check

Status: **NOT REPRODUCED on alternate measured no-retuning surface; inspect measured decision details.**

- diagnostic type: `measured_no_retuning_alternate_surface_against_cached_frozen_gate`
- cached label: `cached_promoted_gate`
- measured label: `measured_alt_surface_len112_cont32`
- method branch: `rdu_topk`

This check reruns the frozen sparse-cache probe with the same `rdu_topk` rule and no policy retuning. Only the measurement surface changes, and the cached first-surface gate is kept as a labeled reference.

## Surface

| Field | Cached | Measured |
|---|---:|---:|
| max_traces | 74 | 74 |
| n_scored_traces | 74 | 73 |
| max_length | 96 | 112 |
| continuation_tokens | 24 | 32 |
| keep_fraction | 0.2 | 0.2 |

## Cached vs Measured Decision

| Label | RDU NLL | Best compressed | Margin vs R-KV | Paired vs R-KV | Margin vs ThinKV | Paired vs ThinKV | RKV/ThinKV rule |
|---|---:|---|---:|---:|---:|---:|---|
| cached_promoted_gate | 3.779 | rdu_topk | +0.160 | -0.160 [-0.264,-0.050] | +0.121 | -0.121 [-0.211,-0.037] | pass |
| measured_alt_surface_len112_cont32 | 3.594 | tf_sparse_r0.55_p0.05_m0.12_a2 | +0.087 | -0.087 [-0.139,-0.028] | +0.256 | -0.256 [-0.465,-0.086] | pass |

## Measured Policy Table

| Policy | NLL | Delta measured-cached |
|---|---:|---:|
| full_cache | 2.747 | -0.101 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 3.588 | -0.320 |
| rdu_topk | 3.594 | -0.185 |
| rkv_like | 3.681 | -0.258 |
| thoughtflow_saliency_recent | 3.694 | -0.226 |
| thin_kv_like | 3.851 | -0.049 |
| longflow_like | 3.881 | -0.277 |

## Strict Separation

Positive margins mean the measured row is worse than `rdu_topk`.

| Family | Policy | Margin NLL vs RDU |
|---|---|---:|
| stopped ThoughtFlow family | thoughtflow_saliency_recent | +0.100 |
| stopped ThoughtFlow family | tf_sparse_r0.55_p0.05_m0.12_a2 | -0.006 |
| cross-family baseline | rkv_like | +0.087 |
| cross-family baseline | thin_kv_like | +0.256 |
| cross-family baseline | longflow_like | +0.287 |

## Oracle And Headroom

- measured per-trace compressed oracle NLL: 3.460
- measured `rdu_topk` NLL: 3.594
- measured full-cache NLL: 2.747
- `rdu_topk` gap to per-trace compressed oracle: 0.135
- per-trace compressed oracle gap to full cache: 0.713
- `rdu_topk` oracle hit rate: 0.438

## Decision

- same-family positive separation: False (min margin -0.006)
- cross-family positive separation: True (min margin +0.087)

This is an alternate measured reproduction surface, not a policy-tuning surface. The next gate is a larger frozen slice or an independently seeded trace split with the same reporting.
