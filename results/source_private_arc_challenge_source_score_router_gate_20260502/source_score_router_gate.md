# ARC Source-Score Router Gate

- date: `2026-05-02`
- pass gate: `False`
- selected metric: `source_index_pair_lookup`
- selected kind: `categorical_lookup`
- test disagreement rows: `473`
- source-score sidecar bytes: `1`

| Row | Accuracy | Delta vs Qwen | CI95 low vs Qwen | Alt-rate |
|---|---:|---:|---:|---:|
| TinyLlama packet | 0.269 | -0.048 | - | - |
| Qwen-substituted packet | 0.317 | 0.000 | - | - |
| source-score router | 0.315 | -0.002 | -0.031 | 0.161 |
| packet oracle | 0.586 | 0.268 | - | - |

## Interpretation

This gate tests whether source-side confidence is enough to repair the strict TinyLlama-vs-Qwen ARC disagreement failure. A positive result would promote risk-gated packet routing as a source-family repair. A negative result means the next branch should be a learned connector or a stronger alternate source, not another scalar confidence rule.

Lay description: we let the source models attach how confident they were in their chosen answer, then trained a tiny validation-only rule to decide which packet to trust. This tests whether the previous failure was just missing source confidence or whether the packet itself needs a learned connector.
