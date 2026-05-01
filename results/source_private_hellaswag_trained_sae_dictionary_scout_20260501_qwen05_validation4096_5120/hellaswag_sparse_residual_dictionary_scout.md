# HellaSwag Sparse Residual Dictionary Scout

- scout pass: `False`
- eval rows: `1024`
- best variant: `sae64_saedecision_relu_top8_dw0p2_l10p001`
- best accuracy: `0.499023`
- best label-copy accuracy: `0.500000`
- delta vs best label-copy: `-0.000977`
- CI95 vs best label-copy: `[-0.015625, 0.011719]`
- score-only bagged control: `0.497070`
- dense hidden-innovation reference: `0.503125`
- packet: `2B` raw / `5B` framed
- dictionary public/preloaded: `True`
- selected-variant dictionary residency: `2.032505` MiB
- native kernel status: `mac_python_trace_only`

## Interpretation

This is the bounded follow-up after PCA learned residual bases passed one slice but failed the five-slice gate. It learns a train-only sparse residual dictionary from candidate residuals and uses sparse atom activations only inside the sender-side packet selector. Success would promote a predeclared all-slice sparse dictionary gate; failure would push us toward a genuinely trained SAE/crosscoder objective rather than another unsupervised codebook.
