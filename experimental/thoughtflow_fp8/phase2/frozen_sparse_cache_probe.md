# ThoughtFlow-FP8 Frozen Sparse-Cache Probe

> Superseded artifact: this historical first-surface positive is preserved for
> auditability only. It is superseded by
> `current_decision_manifest_20260506.md`, which stops the current
> ThoughtFlow-FP8 positive-method branch after stricter reproduction and fresh
> successor gates.

Status: **ALIVE on frozen sparse-cache probe; rdu_topk clears the preregistered promotion rule with margins 0.160 vs R-KV-like and 0.121 vs ThinKV-like.**

- model: `distilgpt2`
- scored traces: 74
- keep fraction: 0.20
- continuation tokens: 24
- frozen ThoughtFlow policies: `thoughtflow_saliency_recent, tf_sparse_r0.55_p0.05_m0.12_a2, rdu_topk`

This larger slice freezes the stopped ThoughtFlow candidates plus the one pre-registered `rdu_topk` successor and performs no policy selection or retuning.
The model processes the full prefix once per trace, prunes the returned KV cache, and scores the continuation from the sparse cache.

| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |
|---|---:|---:|---:|---:|
| full_cache | 74 | 1.000 | 2.848 | 0.000 |
| rdu_topk | 74 | 0.213 | 3.779 | 0.931 |
| thin_kv_like | 74 | 0.213 | 3.900 | 1.052 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 74 | 0.213 | 3.908 | 1.060 |
| thoughtflow_saliency_recent | 74 | 0.213 | 3.920 | 1.072 |
| rkv_like | 74 | 0.214 | 3.939 | 1.091 |
| longflow_like | 74 | 0.213 | 4.158 | 1.311 |

## Paired Delta vs R-KV-like

Negative means lower continuation NLL than rkv_like on the same trace.

| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|
| full_cache | 74 | -1.091 | -1.254 | -0.938 |
| rdu_topk | 74 | -0.160 | -0.264 | -0.050 |
| thin_kv_like | 74 | -0.039 | -0.100 | +0.015 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 74 | -0.031 | -0.078 | +0.020 |
| thoughtflow_saliency_recent | 74 | -0.019 | -0.048 | +0.006 |
| longflow_like | 74 | +0.219 | +0.136 | +0.308 |

## Paired Delta vs ThinKV-like

Negative means lower continuation NLL than thin_kv_like on the same trace.

| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|
| full_cache | 74 | -1.052 | -1.199 | -0.919 |
| rdu_topk | 74 | -0.121 | -0.211 | -0.037 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 74 | +0.008 | -0.060 | +0.085 |
| thoughtflow_saliency_recent | 74 | +0.020 | -0.030 | +0.074 |
| rkv_like | 74 | +0.039 | -0.015 | +0.099 |
| longflow_like | 74 | +0.258 | +0.178 | +0.337 |

## RDU Top-K Telemetry

Telemetry is reported after selection and is not used by the policy.

### Label Retention

| Label | Total | Retained | Retention rate |
|---|---:|---:|---:|
| anchor | 296 | 209 | 0.706 |
| phase | 256 | 8 | 0.031 |
| math_state | 143 | 31 | 0.217 |

### Recurrence-Distance Buckets

| Primary bucket | Tokens | Retained | Retention rate |
|---|---:|---:|---:|
| b0_8_15 | 1652 | 374 | 0.226 |
| b1_16_31 | 287 | 163 | 0.568 |
| b2_32_63 | 2 | 1 | 0.500 |
| b3_64_inf | 0 | 0 | 0.000 |
| none | 592 | 0 | 0.000 |

### Score Summary

| Mean per-trace RDU | Max RDU | Nonzero-token count |
|---:|---:|---:|
| 0.910 | 21.056 | 1941 |

## Decision

Promote `rdu_topk` only if it beats both R-KV-like and ThinKV-like by at least 0.03 NLL with paired CIs below zero.
