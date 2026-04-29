# Source-Private Candidate-Embedding Receiver

- pass gate: `True`
- train/eval: `all:768` / `all:512`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.748 | 0.250 | 0.262 | 0.498 | 0.486 | 0.998 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a trained candidate scorer decodes the packet using public candidate side information.
