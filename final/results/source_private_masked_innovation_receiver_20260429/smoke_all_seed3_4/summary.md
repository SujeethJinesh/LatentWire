# Source-Private Masked Innovation Receiver

- pass gate: `True`
- train/eval: `all:128` / `all:64`
- anchor count: `64`
- source top-k: `48`
- target top-k: `24`
- mask repeats: `1`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.766 | 0.250 | 0.281 | 0.516 | 0.484 | 1.000 |
| 8 | `True` | 0.922 | 0.250 | 0.266 | 0.672 | 0.656 | 1.000 |

## Interpretation

This gate transmits a sparse source-private innovation: hashed matched-source minus answer-masked-source features are mapped to anchor-relative target innovation from target prior candidate to answer candidate. It tests whether private source evidence survives without raw candidate-coordinate regression.
