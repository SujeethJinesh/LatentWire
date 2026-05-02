# ARC Source-Family/Source-Cache Falsification Gate

- date: `2026-05-02`
- pass gate: `False`
- alternate source family: `tinyllama_1.1b`
- packet budget: `12B`
- test full-slice pass: `True`
- test Qwen-disagreement pass: `False`
- test Qwen disagreement rows: `473`

| Split | Surface | Pass seeds | Matched mean | Target | Text | Qwen-sub | Min CI target | Min CI Qwen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | full | 3/5 | 0.320 | 0.244 | 0.298 | - | -0.003 | - |
| validation | Qwen-disagreement | 0/5 | 0.250 | 0.271 | 0.236 | 0.389 | -0.132 | -0.281 |
| test | full | 5/5 | 0.325 | 0.265 | 0.298 | - | 0.018 | - |
| test | Qwen-disagreement | 0/5 | 0.269 | 0.268 | 0.258 | 0.317 | -0.059 | -0.118 |

## Interpretation

This gate directly tests whether the ARC Fourier/anchor-syndrome result depends on the original Qwen source-choice cache. A pass would promote the method to source-family-general packet communication. A failure is still useful: it says the current ARC positive row remains source-cache specific and must be framed below ICLR headline strength until a stronger cross-family source endpoint lands.

Lay description: we ask a different local source model to choose answers, encode those choices with the same tiny Fourier/anchor packet, and then focus on examples where the Qwen source chose something else. If the alternate packet still wins there, the result is less likely to be a Qwen cache artifact.
