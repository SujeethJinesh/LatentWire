# ThoughtFlow-FP8 Hidden-Saliency Retention Probe

Status: **MIXED; thoughtflow beats value_norm_topk on phase recall but does not clear the full proxy set.**

- model: `distilgpt2`
- traces: 24
- keep fraction: 0.20

This uses real distilgpt2 attention, final-hidden norms, key norms, value norms, and combined KV norms as CPU hidden/KV saliency proxies.
It is not a cache-compression accuracy result and not a GPU benchmark.
- best ThoughtFlow-family policy: `thoughtflow`
- strongest real-saliency proxy: `value_norm_topk`
- phase margin vs real saliency: +0.508
- math-state margin vs real saliency: +0.073

| Policy | Keep rate | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|---:|
| attention_received_topk | 0.207 | 1.000 | 0.087 | 0.720 |
| hidden_norm_topk | 0.207 | 0.010 | 0.532 | 0.722 |
| key_norm_topk | 0.207 | 0.000 | 0.300 | 0.828 |
| kv_norm_topk | 0.207 | 0.000 | 0.411 | 0.845 |
| longflow_like | 0.207 | 1.000 | 1.000 | 0.918 |
| rkv_like | 0.207 | 1.000 | 0.083 | 0.747 |
| thin_kv_like | 0.207 | 1.000 | 0.948 | 0.848 |
| thoughtflow | 0.207 | 1.000 | 1.000 | 0.918 |
| thoughtflow_saliency_recent | 0.207 | 1.000 | 0.714 | 0.726 |
| thoughtflow_sweep_best | 0.207 | 1.000 | 0.430 | 0.956 |
| value_norm_topk | 0.207 | 0.000 | 0.492 | 0.845 |

## Paired Margins

Mean paired recall delta for the best ThoughtFlow-family policy minus the strongest real-saliency proxy.

| Metric | Traces | Mean delta | 95% CI |
|---|---:|---:|---:|
| anchor_recall | 24 | +1.000 | [+1.000, +1.000] |
| phase_recall | 24 | +0.508 | [+0.436, +0.579] |
| math_state_recall | 24 | +0.073 | [-0.078, +0.223] |

## Decision

Advance ThoughtFlow only if a train-fixed ThoughtFlow-family policy beats real hidden/KV saliency proxies on both phase/control and math-state recall.
A phase-only win is not enough; it can be a marker-preservation artifact rather than useful cache retention.
