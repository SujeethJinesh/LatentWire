# ARC Source-Family/Source-Cache Falsification Gate

- date: `2026-05-02`
- pass gate: `False`
- alternate source family: `qwen2.5_1.5b`
- packet budget: `12B`
- test full-slice pass: `True`
- test Qwen-disagreement pass: `True`
- test Qwen disagreement rows: `388`

| Split | Surface | Pass seeds | Matched mean | Target | Text | Qwen-sub | Min CI target | Min CI Qwen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | full | 5/5 | 0.437 | 0.244 | 0.371 | - | 0.114 | - |
| validation | Qwen-disagreement | 0/5 | 0.389 | 0.211 | 0.326 | 0.232 | 0.042 | -0.021 |
| test | full | 5/5 | 0.442 | 0.265 | 0.401 | - | 0.137 | - |
| test | Qwen-disagreement | 5/5 | 0.482 | 0.296 | 0.456 | 0.184 | 0.112 | 0.216 |

## Interpretation

This gate directly tests whether the ARC Fourier/anchor-syndrome result depends on the original Qwen source-choice cache. A pass would promote the method to source-family-general packet communication. A failure is still useful: it says the current ARC positive row remains source-cache specific and must be framed below ICLR headline strength until a stronger cross-family source endpoint lands.

Lay description: we ask a different local source model to choose answers, encode those choices with the same tiny Fourier/anchor packet, and then focus on examples where the Qwen source chose something else. If the alternate packet still wins there, the result is less likely to be a Qwen cache artifact.
