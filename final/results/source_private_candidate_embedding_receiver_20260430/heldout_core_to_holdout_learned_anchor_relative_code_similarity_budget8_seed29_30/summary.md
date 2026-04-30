# Source-Private Candidate-Embedding Receiver

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate feature dims: `0`
- receiver kind: `code_similarity`
- packet feature mode: `learned_anchor_relative`
- packet dim: `128`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `False` | 0.250 | 0.250 | 0.268 | 0.000 | -0.018 | 0.723 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a target-side scorer decodes the packet using public candidate side information.
