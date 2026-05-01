# HellaSwag Dense Residual Sketch Scout

- scout pass: `False`
- eval rows: `1024`
- best variant: `qjl_norm_sign128`
- best accuracy: `0.501953`
- best label-copy accuracy: `0.500000`
- delta vs best label-copy: `0.001953`
- CI95 vs best label-copy: `[-0.011719, 0.015161]`
- score-only bagged control: `0.497070`
- dense hidden-innovation reference: `0.503125`
- packet: `2B` raw / `5B` framed

## Interpretation

This is a bounded rescue scout after anchor/common-basis variants failed. It tests whether public random projections or 1-bit sign sketches of dense source hidden residuals preserve enough private decision evidence to select the same tiny source-private packet. Because variants are compared on one eval slice, success would only promote a predeclared all-slice gate; failure pushes the next branch toward learned sparse/crosscoder-style dictionaries.
