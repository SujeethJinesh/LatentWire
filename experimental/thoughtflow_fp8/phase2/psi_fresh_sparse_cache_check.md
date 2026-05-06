# ThoughtFlow-FP8 Prefix-Surprisal Fresh Sparse-Cache Check

Status: **KILLED on one-shot fresh sparse-cache surface; psi_topk fails the preregistered promotion rule.**

- model: `distilgpt2`
- policy: `psi_topk`
- scored traces: 70
- keep fraction: 0.20
- max length: 96
- continuation tokens: 24

This is a one-shot run of the pre-registered prefix-surprisal utility on a saved-trace surface not used by the RDU promotion gate. It performs no sweep, retuning, or variant selection.

## Trace Inputs

- `results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl`

## Policy Table

| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |
|---|---:|---:|---:|---:|
| full_cache | 70 | 1.000 | 3.041 | 0.000 |
| thin_kv_like | 70 | 0.216 | 3.906 | 0.864 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 70 | 0.216 | 3.907 | 0.866 |
| rkv_like | 70 | 0.220 | 3.960 | 0.919 |
| thoughtflow_saliency_recent | 70 | 0.216 | 3.967 | 0.926 |
| longflow_like | 70 | 0.216 | 4.119 | 1.078 |
| psi_topk | 70 | 0.216 | 7.899 | 4.858 |

## Promotion Readout

- best compressed policy: `thin_kv_like`
- `psi_topk` margin vs R-KV-like: -3.939
- `psi_topk` paired delta vs R-KV-like: +3.939 [+3.720,+4.144]
- `psi_topk` margin vs ThinKV-like: -3.994
- `psi_topk` paired delta vs ThinKV-like: +3.994 [+3.773,+4.205]
- promotion pass: False

## Prefix-Surprisal Telemetry

| Mean surprisal | Mean kept surprisal | Max surprisal | Nonzero tokens |
|---:|---:|---:|---:|
| 4.320 | 9.564 | 18.589 | 1643 |

## Decision

Promotion required `psi_topk` to beat both R-KV-like and ThinKV-like by at least 0.03 NLL with paired CIs below zero and to be the best compressed row. Failure rules out this exact signal in the current Mac sparse-cache harness.
