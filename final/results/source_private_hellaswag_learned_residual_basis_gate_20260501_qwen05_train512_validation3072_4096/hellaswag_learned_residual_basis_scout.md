# HellaSwag Learned Residual Basis Scout

- scout pass: `True`
- eval rows: `1024`
- best variant: `pca256_top2_gold`
- best accuracy: `0.504883`
- best label-copy accuracy: `0.484375`
- delta vs best label-copy: `0.020508`
- CI95 vs best label-copy: `[0.003906, 0.038086]`
- score-only bagged control: `0.480469`
- dense hidden-innovation reference: `0.503125`
- packet: `2B` raw / `5B` framed

## Interpretation

This is a bounded scout after anchor charts and random sign sketches failed. It tests whether a train-only learned residual basis, used only inside the sender-side packet selector, preserves the dense hidden-innovation signal. This is a cheap proxy for a learned sparse/crosscoder dictionary; success would promote a predeclared all-slice basis gate, while failure would require a richer sparse/SAE-style training objective rather than PCA.
