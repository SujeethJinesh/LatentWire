# Toy Verifier-Guided Atom Pruning

- Seed: `0`
- Train examples: `160`
- Test examples: `128`
- Atoms: `18`
- Steps: `3`
- Keep fraction: `0.5`

| Method | Accuracy | MSE | Prune rate | Missed help | False prune | Atom recovery | Bytes proxy | Compute proxy | Help vs no-pruning | Harm vs no-pruning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_pruning | 0.8047 | 0.1083 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1152.0000 | 743.4484 | 0.0000 | 0.0000 |
| scalar_score_pruning | 0.5234 | 0.1826 | 0.5143 | 0.4696 | 0.4555 | 0.5321 | 559.5000 | 358.0539 | 0.0000 | 0.2812 |
| step_error_localized_pruning | 0.9062 | 0.1068 | 0.5013 | 0.3715 | 0.3705 | 0.6250 | 574.5000 | 357.1310 | 0.1016 | 0.0000 |
| verifier_guided_frontier_pruning | 0.9609 | 0.0601 | 0.5009 | 0.2726 | 0.2720 | 0.7240 | 575.0000 | 354.3978 | 0.1562 | 0.0000 |
| oracle_pruning | 0.8984 | 0.0190 | 0.5000 | 0.0095 | 0.0095 | 1.0000 | 576.0000 | 362.0784 | 0.0938 | 0.0000 |
