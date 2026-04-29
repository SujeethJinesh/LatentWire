# Source-Private Candidate-Embedding Receiver

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `False` | 0.453 | 0.250 | 0.311 | 0.203 | 0.143 | 0.809 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a trained candidate scorer decodes the packet using public candidate side information.
