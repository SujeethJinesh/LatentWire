# ThoughtFlow-FP8 CPU Sparse-KV Drop Quality Probe

Status: **MIXED on CPU sparse-KV probe; thoughtflow_saliency_recent ties thin_kv_like within 0.03 NLL.**

- model: `distilgpt2`
- scored traces: 24
- keep fraction: 0.20
- continuation tokens: 24

This probe runs the full prefix once, prunes the returned KV cache according to each policy, and scores the continuation from the pruned cache.
It is CPU-only quality evidence, not a Triton/CUDA performance result.

| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |
|---|---:|---:|---:|---:|
| full_cache | 24 | 1.000 | 2.142 | 0.000 |
| thoughtflow_saliency_recent | 24 | 0.210 | 3.372 | 1.230 |
| thin_kv_like | 24 | 0.210 | 3.389 | 1.247 |
| thoughtflow_recent | 24 | 0.210 | 3.399 | 1.257 |
| thoughtflow_sweep_best | 24 | 0.210 | 3.436 | 1.294 |
| rkv_like | 24 | 0.210 | 3.438 | 1.296 |
| longflow_like | 24 | 0.210 | 3.588 | 1.446 |
| thoughtflow | 24 | 0.210 | 3.588 | 1.446 |

## Paired Delta vs R-KV-like

Negative means lower continuation NLL than R-KV-like on the same trace.

| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|
| full_cache | 24 | -1.296 | -1.533 | -1.066 |
| thoughtflow_saliency_recent | 24 | -0.067 | -0.151 | +0.011 |
| thin_kv_like | 24 | -0.049 | -0.192 | +0.077 |
| thoughtflow_recent | 24 | -0.040 | -0.169 | +0.074 |
| thoughtflow_sweep_best | 24 | -0.002 | -0.007 | +0.000 |
| longflow_like | 24 | +0.150 | -0.016 | +0.300 |
| thoughtflow | 24 | +0.150 | -0.006 | +0.297 |

## Paired Delta vs ThinKV-like

Negative means lower continuation NLL than ThinKV-like on the same trace.

| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|
| full_cache | 24 | -1.247 | -1.418 | -1.061 |
| thoughtflow_saliency_recent | 24 | -0.018 | -0.104 | +0.072 |
| thoughtflow_recent | 24 | +0.010 | -0.037 | +0.056 |
| thoughtflow_sweep_best | 24 | +0.047 | -0.082 | +0.186 |
| rkv_like | 24 | +0.049 | -0.078 | +0.191 |
| longflow_like | 24 | +0.199 | +0.105 | +0.295 |
| thoughtflow | 24 | +0.199 | +0.104 | +0.296 |

## Train-Fixed Sparse Sweep

Status: **MIXED on train-fixed sparse sweep; held-out policy clears mean margin but not paired uncertainty.**

- train traces: 12
- held-out traces: 12
- best train-selected policy: `tf_sparse_r0.55_p0.05_m0.12_a2`
- strongest non-ThoughtFlow held-out baseline: `thin_kv_like`
- held-out NLL margin vs strongest non-ThoughtFlow baseline: +0.045

| Policy | Held-out traces | Keep rate | NLL |
|---|---:|---:|---:|
| thoughtflow_saliency_recent | 12 | 0.211 | 3.304 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 12 | 0.211 | 3.340 |
| thin_kv_like | 12 | 0.211 | 3.385 |
| rkv_like | 12 | 0.211 | 3.420 |

| Policy | Baseline | Mean delta NLL | 95% CI low | 95% CI high |
|---|---|---:|---:|---:|
| thoughtflow_saliency_recent | rkv_like | -0.116 | -0.252 | +0.002 |
| thoughtflow_saliency_recent | thin_kv_like | -0.081 | -0.212 | +0.057 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | rkv_like | -0.080 | -0.152 | -0.014 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | thin_kv_like | -0.045 | -0.226 | +0.182 |

## Decision

This is the closest Mac gate to actual cache dropping. Advance only if a train-fixed ThoughtFlow-family policy beats R-KV-like and ThinKV-like on matched-budget continuation NLL with paired uncertainty.
