# ARC Source-Family/Source-Cache Falsification Gate

- date: `2026-05-02`
- pass gate: `False`
- alternate source family: `phi3_mini_4k`
- packet budget: `12B`
- test full-slice pass: `False`
- test Qwen-disagreement pass: `False`
- test Qwen disagreement rows: `833`

| Split | Surface | Pass seeds | Matched mean | Target | Text | Qwen-sub | Min CI target | Min CI Qwen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | full | 0/5 | 0.278 | 0.244 | 0.268 | - | -0.047 | - |
| validation | Qwen-disagreement | 0/5 | 0.238 | 0.229 | 0.238 | 0.384 | -0.079 | -0.256 |
| test | full | 0/5 | 0.244 | 0.265 | 0.241 | - | -0.058 | - |
| test | Qwen-disagreement | 0/5 | 0.200 | 0.273 | 0.209 | 0.340 | -0.116 | -0.193 |

## Interpretation

This gate directly tests whether the ARC Fourier/anchor-syndrome result depends on the original Qwen source-choice cache. A pass would promote the method to source-family-general packet communication. A failure is still useful: it says the current ARC positive row remains source-cache specific and must be framed below ICLR headline strength until a stronger cross-family source endpoint lands.

Lay description: we ask a different local source model to choose answers, encode those choices with the same tiny Fourier/anchor packet, and then focus on examples where the Qwen source chose something else. If the alternate packet still wins there, the result is less likely to be a Qwen cache artifact.
