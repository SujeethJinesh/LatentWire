# Cross-Layer Quantization Error Compounding Derivation

Locked before measurement at `2026-05-08T19:16:15Z`.

This packet uses only BF16 model weights and the fixed depth pattern to compute the bound.
No BF16-vs-FP4 logit drift rows are read before this document and `predicted_bounds.json` are written.

For each quantized layer `l`, each floating-point parameter tensor is block-quantized with the
repo-local E2M1 block-scaled FP4 simulator (`nvfp4_e2m1_weight_sim`, block size 32). Let
`e_l = Q_l(W_l) - W_l`. The recorded variance terms are:

- `sigma_block_l = mean(e_l^2)` over all layer parameters.
- `sigma_outlier_l = mean(e_l^2)` over the top 1% absolute-weight entries.
- `eta_l = sqrt(sum(e_l^2) + sum(e_l,outlier^2))`.

The output map is bounded by `C_out = ||W_lm_head||_F / sqrt(hidden_size)` when an LM head is present,
falling back to `1.0` only if the output embedding cannot be resolved.

For a depth pattern that quantizes the first `N` layers, the preregistered prediction used here is:

`F(N, sigma_block, sigma_outlier, depth_pattern) = C_out * sqrt(sum_{l in first N layers} eta_l^2)`

This is a first-principles Lipschitz-style upper-bound attempt. It is not fit to measured drift.

- Output-scale source: `lm_head_frobenius_norm_over_sqrt_hidden`
- Output scale: `9.72966322326`
- Vocab size: `100352`
- Hidden size: `1536`

| Depth | Predicted F | sigma_block_sum | sigma_outlier_sum | layer_error_l2_quadrature |
|---:|---:|---:|---:|---:|
| 1 | 1323.28194307 | 0.000104026583533 | 0.000440628819484 | 136.004907128 |
| 5 | 2011.75966405 | 0.000241233625627 | 0.000939786497018 | 206.765601017 |
| 10 | 2669.6676375 | 0.0004267404374 | 0.00148157050144 | 274.384382711 |
| 15 | 3267.82871155 | 0.000641068825561 | 0.00204608526504 | 335.862468882 |
