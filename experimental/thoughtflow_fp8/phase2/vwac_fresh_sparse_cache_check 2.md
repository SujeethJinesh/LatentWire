# ThoughtFlow-FP8 Value-Weighted Attention Contribution Fresh Check

Status: **KILLED on one-shot fresh sparse-cache surface; vwac_topk fails the preregistered promotion rule.**

- model: `distilgpt2`
- policy: `vwac_topk`
- scored traces: 64
- keep fraction: 0.20
- max length: 96
- continuation tokens: 24

This is a one-shot run of the pre-registered value-weighted attention-contribution utility on a saved-trace surface not used by the RDU or PSI promotion gates.

## Trace Inputs

- `results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl`

## Policy Table

| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |
|---|---:|---:|---:|---:|
| full_cache | 64 | 1.000 | 3.269 | 0.000 |
| rkv_like | 64 | 0.239 | 4.096 | 0.827 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 64 | 0.222 | 4.130 | 0.861 |
| thoughtflow_saliency_recent | 64 | 0.222 | 4.138 | 0.869 |
| thin_kv_like | 64 | 0.222 | 4.162 | 0.893 |
| longflow_like | 64 | 0.222 | 4.255 | 0.987 |
| vwac_topk | 64 | 0.222 | 4.336 | 1.068 |

## Promotion Readout

- best compressed policy: `rkv_like`
- `vwac_topk` margin vs R-KV-like: -0.241
- `vwac_topk` paired delta vs R-KV-like: +0.241 [+0.030,+0.530]
- `vwac_topk` margin vs ThinKV-like: -0.174
- `vwac_topk` paired delta vs ThinKV-like: +0.174 [+0.002,+0.392]
- promotion pass: False

## VWAC Telemetry

| Mean VWAC | Mean kept VWAC | Max VWAC | Nonzero tokens |
|---:|---:|---:|---:|
| 4.813 | 7.764 | 13.728 | 1396 |
