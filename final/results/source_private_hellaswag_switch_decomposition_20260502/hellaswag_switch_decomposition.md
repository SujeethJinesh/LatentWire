# HellaSwag Switch Decomposition

- pass gate: `False`
- selected train-dev switch view: `hidden_score_switch`
- validation[0:1024] selected accuracy: `0.449219`
- validation[0:1024] selected delta vs best label-copy: `-0.012695`
- validation[0:1024] selected switch precision: `0.3333333333333333`
- validation[0:1024] top-2 oracle accuracy: `0.715820`
- terminal-tail selected accuracy: `0.485472`
- terminal-tail selected delta vs best label-copy: `-0.012107`
- terminal-tail selected switch precision: `0.3531746031746032`
- terminal-tail top-2 oracle accuracy: `0.756659`

## Interpretation

This decomposition isolates whether HellaSwag gains come from a deployable top-2 switch decision. A switch-only method is publication-worthy only if hidden-private evidence chooses source top-2 with high precision, beats score-only switching, and survives wrong-hidden and label-permuted controls on both the passed validation-first1024 slice and the fragile terminal tail.
