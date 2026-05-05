# ThoughtFlow-FP8 Held-Out Policy Sweep

Status: **MIXED on held-out policy sweep; best ThoughtFlow policy ties R-KV-like within 0.03 NLL.**

- model: `distilgpt2`
- train traces: 12
- held-out traces: 12
- keep fraction: 0.20
- best train-selected policy: `tf_sweep_r0.55_p0.00_m0.18_a4`
- held-out NLL margin vs R-KV-like: +0.001 (positive means ThoughtFlow is better)

## Held-Out Summary

| Policy | Traces | Keep rate | NLL | PPL |
|---|---:|---:|---:|---:|
| tf_sweep_r0.55_p0.00_m0.18_a4 | 12 | 0.211 | 3.480 | 43.7 |
| rkv_like | 12 | 0.211 | 3.482 | 43.7 |
| thoughtflow_saliency_recent | 12 | 0.211 | 3.497 | 42.3 |
| thin_kv_like | 12 | 0.211 | 3.674 | 50.7 |

## Decision

This is still a text-prefix proxy, not sparse-KV decoding. A positive workshop method requires the train-selected ThoughtFlow-family policy to beat R-KV-like on held-out continuation NLL, then validate under real hidden/KV telemetry.
