# Source-Private Candidate-Embedding Receiver

- pass gate: `False`
- train/eval: `core:768` / `holdout:512`
- candidate feature dims: `0`
- receiver kind: `ridge`
- packet feature mode: `anchor_relative`
- packet dim: `128`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `False` | 0.303 | 0.250 | 0.438 | 0.053 | -0.135 | 0.342 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a target-side scorer decodes the packet using public candidate side information.
