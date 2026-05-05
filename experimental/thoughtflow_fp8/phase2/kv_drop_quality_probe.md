# ThoughtFlow-FP8 CPU Sparse-KV Drop Quality Probe

Status: **MIXED on CPU sparse-KV probe; thoughtflow_sweep_best ties rkv_like within 0.03 NLL.**

- model: `distilgpt2`
- scored traces: 24
- keep fraction: 0.20
- continuation tokens: 16

This probe runs the full prefix once, prunes the returned KV cache according to each policy, and scores the continuation from the pruned cache.
It is CPU-only quality evidence, not a Triton/CUDA performance result.

| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |
|---|---:|---:|---:|---:|
| full_cache | 24 | 1.000 | 2.085 | 0.000 |
| thoughtflow_sweep_best | 24 | 0.210 | 3.432 | 1.347 |
| rkv_like | 24 | 0.210 | 3.435 | 1.350 |
| thoughtflow_saliency_recent | 24 | 0.210 | 3.488 | 1.403 |
| thin_kv_like | 24 | 0.210 | 3.624 | 1.539 |
| thoughtflow_recent | 24 | 0.210 | 3.648 | 1.563 |
| longflow_like | 24 | 0.210 | 3.782 | 1.697 |
| thoughtflow | 24 | 0.210 | 3.782 | 1.697 |

## Paired Delta vs R-KV-like

Negative means lower continuation NLL than R-KV-like on the same trace.

| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|
| full_cache | 24 | -1.350 | -1.618 | -1.098 |
| thoughtflow_sweep_best | 24 | -0.003 | -0.037 | +0.034 |
| thoughtflow_saliency_recent | 24 | +0.053 | -0.011 | +0.123 |
| thin_kv_like | 24 | +0.189 | +0.047 | +0.330 |
| thoughtflow_recent | 24 | +0.213 | +0.109 | +0.324 |
| longflow_like | 24 | +0.348 | +0.159 | +0.552 |
| thoughtflow | 24 | +0.348 | +0.141 | +0.548 |

## Decision

This is the closest Mac gate to actual cache dropping. Advance only if a train-fixed ThoughtFlow-family policy beats R-KV-like and ThinKV-like on matched-budget continuation NLL with paired uncertainty.
