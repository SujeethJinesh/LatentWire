# Source-Private Candidate-Embedding Receiver

- pass gate: `False`
- train/eval: `all:768` / `all:512`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.328 | 0.250 | 0.279 | 0.078 | 0.049 | 1.000 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a trained candidate scorer decodes the packet using public candidate side information.
