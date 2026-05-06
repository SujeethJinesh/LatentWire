# ThoughtFlow-FP8 RDU Robustness Diagnostic

> Superseded artifact: this cached split diagnostic is preserved for
> auditability only. It is superseded by
> `current_decision_manifest_20260506.md`, which stops the current
> ThoughtFlow-FP8 positive-method branch after independent reproduction and
> fresh successor gates.

Status: **PROMOTED on cached full gate; deterministic trace splits keep positive margins and paired means, but split CIs are not uniformly below zero.**

- source artifact: `frozen_sparse_cache_probe.json`
- model: `distilgpt2`
- keep fraction: 0.20
- scored traces: 74
- continuation tokens: 24
- bootstrap samples per paired CI: 1000

This diagnostic reuses the cached 0.20 frozen sparse-cache rows. It does not rerun the model, retune a policy, or change the pre-registered `rdu_topk` scoring rule.

| Split | Traces | Best compressed | RDU NLL | Margin vs R-KV | Paired vs R-KV | Win rate vs R-KV | Margin vs ThinKV | Paired vs ThinKV | Win rate vs ThinKV | Promotion rule |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| all_traces | 74 | rdu_topk | 3.779 | +0.160 | -0.160 [-0.264,-0.050] | 0.622 | +0.121 | -0.121 [-0.211,-0.037] | 0.581 | pass |
| even_trace_ids | 37 | rdu_topk | 3.739 | +0.186 | -0.186 [-0.324,-0.071] | 0.649 | +0.101 | -0.101 [-0.198,-0.010] | 0.595 | pass |
| odd_trace_ids | 37 | rdu_topk | 3.818 | +0.134 | -0.134 [-0.314,+0.045] | 0.595 | +0.141 | -0.141 [-0.281,+0.003] | 0.568 | mean-only |
| first_half_trace_ids | 37 | rdu_topk | 3.294 | +0.154 | -0.154 [-0.300,-0.010] | 0.622 | +0.132 | -0.132 [-0.236,-0.013] | 0.595 | pass |
| second_half_trace_ids | 37 | rdu_topk | 4.263 | +0.166 | -0.166 [-0.334,-0.026] | 0.622 | +0.110 | -0.110 [-0.240,+0.003] | 0.568 | mean-only |

## Interpretation

The cached full gate still satisfies the pre-registered promotion rule. All four deterministic half-size diagnostics keep positive mean margins of at least 0.03 NLL versus both R-KV-like and ThinKV-like, and `rdu_topk` remains the best compressed row in each split.

Half-size paired CIs are intentionally treated as a stress diagnostic, not as a replacement for a fresh reproduction. The odd and second-half partitions leave some CI highs slightly above zero, so this result strengthens the branch but does not make it ICLR-ready.

Same-family separation is also preserved on the full cached gate: `rdu_topk` beats the stopped ThoughtFlow candidates by thoughtflow_saliency_recent: 0.141 NLL, tf_sparse_r0.55_p0.05_m0.12_a2: 0.129 NLL.

## Decision

`rdu_topk` remains the best compressed row on the first frozen sparse-cache decision surface. Later alternate-surface and independent-trace gates demote it, so this artifact is historical support only.
