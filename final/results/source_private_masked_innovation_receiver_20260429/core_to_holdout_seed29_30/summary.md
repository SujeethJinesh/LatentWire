# Source-Private Masked Innovation Receiver

- pass gate: `False`
- train/eval: `core:256` / `holdout:128`
- anchor count: `64`
- source top-k: `48`
- target top-k: `24`
- mask repeats: `1`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.258 | 0.250 | 0.258 | 0.008 | 0.000 | 1.000 |
| 8 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 1.000 |
| 12 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 1.000 |

## Interpretation

This gate transmits a sparse source-private innovation: hashed matched-source minus answer-masked-source features are mapped to anchor-relative target innovation from target prior candidate to answer candidate. It tests whether private source evidence survives without raw candidate-coordinate regression.
