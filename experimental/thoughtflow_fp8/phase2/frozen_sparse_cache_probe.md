# ThoughtFlow-FP8 Frozen Sparse-Cache Probe

Status: **MIXED on frozen sparse-cache probe; tf_sparse_r0.55_p0.05_m0.12_a2 remains inside 0.03 NLL vs thin_kv_like.**

- model: `distilgpt2`
- scored traces: 74
- keep fraction: 0.20
- continuation tokens: 24
- frozen ThoughtFlow policies: `thoughtflow_saliency_recent, tf_sparse_r0.55_p0.05_m0.12_a2`

This larger slice freezes the two current ThoughtFlow candidates and performs no policy selection or retuning.
The model processes the full prefix once per trace, prunes the returned KV cache, and scores the continuation from the sparse cache.

| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |
|---|---:|---:|---:|---:|
| full_cache | 74 | 1.000 | 2.848 | 0.000 |
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
| thin_kv_like | 74 | -0.039 | -0.100 | +0.015 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 74 | -0.031 | -0.078 | +0.020 |
| thoughtflow_saliency_recent | 74 | -0.019 | -0.048 | +0.006 |
| longflow_like | 74 | +0.219 | +0.136 | +0.308 |

## Paired Delta vs ThinKV-like

Negative means lower continuation NLL than thin_kv_like on the same trace.

| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|
| full_cache | 74 | -1.052 | -1.199 | -0.919 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 74 | +0.008 | -0.060 | +0.085 |
| thoughtflow_saliency_recent | 74 | +0.020 | -0.030 | +0.074 |
| rkv_like | 74 | +0.039 | -0.015 | +0.099 |
| longflow_like | 74 | +0.258 | +0.178 | +0.337 |

## Decision

Promote only if a frozen ThoughtFlow policy beats both R-KV-like and ThinKV-like by at least 0.03 NLL with paired CIs below zero.
