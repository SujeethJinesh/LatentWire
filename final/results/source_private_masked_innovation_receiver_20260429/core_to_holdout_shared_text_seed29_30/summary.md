# Source-Private Masked Innovation Receiver

- pass gate: `False`
- train/eval: `core:128` / `holdout:64`
- candidate view: `shared_text`
- anchor count: `64`
- representation dim: `128`
- source top-k: `32`
- target top-k: `32`
- mask repeats: `1`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.266 | 0.250 | 0.250 | 0.016 | 0.016 | 1.000 |
| 8 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 1.000 |

## Interpretation

This gate transmits a sparse source-private innovation: hashed matched-source minus answer-masked-source features are mapped to anchor-relative target innovation from target prior candidate to answer candidate. It tests whether private source evidence survives without raw candidate-coordinate regression.
