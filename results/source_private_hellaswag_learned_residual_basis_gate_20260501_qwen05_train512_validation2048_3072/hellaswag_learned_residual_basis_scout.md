# HellaSwag Learned Residual Basis Scout

- scout pass: `False`
- eval rows: `1024`
- best variant: `pca256_top2_gold`
- best accuracy: `0.452148`
- best label-copy accuracy: `0.445312`
- delta vs best label-copy: `0.006836`
- CI95 vs best label-copy: `[-0.013208, 0.025391]`
- score-only bagged control: `0.433594`
- dense hidden-innovation reference: `0.503125`
- packet: `2B` raw / `5B` framed

## Interpretation

This is a bounded scout after anchor charts and random sign sketches failed. It tests whether a train-only learned residual basis, used only inside the sender-side packet selector, preserves the dense hidden-innovation signal. This is a cheap proxy for a learned sparse/crosscoder dictionary; success would promote a predeclared all-slice basis gate, while failure would require a richer sparse/SAE-style training objective rather than PCA.
