# Source-Private Candidate-Embedding Receiver

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate feature dims: `0`
- receiver kind: `code_similarity`
- packet feature mode: `anchor_relative`
- packet dim: `128`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `False` | 0.281 | 0.250 | 0.258 | 0.031 | 0.023 | 0.756 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a target-side scorer decodes the packet using public candidate side information.
