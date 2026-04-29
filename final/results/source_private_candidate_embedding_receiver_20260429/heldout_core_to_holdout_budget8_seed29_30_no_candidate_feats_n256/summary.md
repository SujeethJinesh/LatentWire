# Source-Private Candidate-Embedding Receiver

- pass gate: `False`
- train/eval: `core:768` / `holdout:256`
- candidate feature dims: `0`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `False` | 0.332 | 0.250 | 0.309 | 0.082 | 0.023 | 0.742 |

## Interpretation

This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, and a trained candidate scorer decodes the packet using public candidate side information.
