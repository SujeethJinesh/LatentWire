# ThoughtFlow-FP8 Retained-Context Perplexity Proxy

Status: **WEAKENED on retained-context NLL proxy; matched-budget proxies score continuation better.**

- model: `distilgpt2`
- scored traces: 24
- keep fraction: 0.20
- max length: 96
- continuation tokens: 24

This is a Mac-local quality proxy, not sparse-KV decoding. It compresses
the trace prefix, appends the same held-out continuation, and scores only
continuation NLL. Full context is a reference row, not a matched-budget baseline.

| Policy | Traces | Keep rate | NLL | Delta NLL vs full | PPL |
|---|---:|---:|---:|---:|---:|
| full_context | 24 | 1.000 | 2.101 | 0.000 | 9.7 |
| longflow_like | 24 | 0.210 | 3.961 | 1.861 | 74.2 |
| rkv_like | 24 | 0.210 | 3.419 | 1.319 | 38.7 |
| thin_kv_like | 24 | 0.210 | 3.583 | 1.482 | 44.6 |
| thoughtflow | 24 | 0.210 | 3.961 | 1.861 | 74.2 |

## Decision

Advance only if ThoughtFlow beats all matched-budget compressed proxies on continuation NLL.
A tie or loss keeps the current branch mixed/weakened and defers GPU/KV work until a sharper policy exists.
