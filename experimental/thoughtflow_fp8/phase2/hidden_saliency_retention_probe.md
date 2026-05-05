# ThoughtFlow-FP8 Hidden-Saliency Retention Probe

Status: **MIXED; beats attention-saliency proxy but still ties the strongest importance proxy.**

- model: `distilgpt2`
- traces: 24
- keep fraction: 0.20

This uses attention-received mass as a CPU hidden/KV saliency proxy.
It is not a cache-compression accuracy result and not a GPU benchmark.

| Policy | Keep rate | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|---:|
| attention_received_topk | 0.207 | 0.979 | 0.299 | 0.039 |
| longflow_like | 0.207 | 1.000 | 1.000 | 0.925 |
| rkv_like | 0.207 | 1.000 | 0.000 | 0.134 |
| thin_kv_like | 0.207 | 1.000 | 0.979 | 0.813 |
| thoughtflow | 0.207 | 1.000 | 1.000 | 0.925 |

## Decision

Advance ThoughtFlow only if the protected-token policy beats a hidden/KV saliency proxy.
If it loses or ties, the current phase-marker heuristic should not go to GPU work.
