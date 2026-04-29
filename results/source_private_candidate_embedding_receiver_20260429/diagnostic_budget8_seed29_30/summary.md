# Source-Private Candidate-Embedding Receiver

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `True` | 0.875 | 0.250 | 0.250 | 0.625 | 0.625 | 1.000 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a trained candidate scorer decodes the packet using public candidate side information.
