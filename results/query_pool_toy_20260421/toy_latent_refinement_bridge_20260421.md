# Toy Latent Refinement Bridge

| Method | Accuracy | MSE | Entropy | Confidence | Bytes estimate | Steps |
|---|---:|---:|---:|---:|---:|---:|
| one_shot_noisy_bridge | 0.7604 | 0.3274 | 1.3424 | 0.4461 | 16.0000 | 1.0000 |
| iterative_residual_refinement | 0.9792 | 0.0126 | 1.3292 | 0.4581 | 39.0000 | 3.0000 |
| gated_refinement | 0.9271 | 0.0553 | 1.3282 | 0.4559 | 17.0000 | 2.0000 |
| soft_token_mixture_projection | 0.5417 | 0.8733 | 1.5752 | 0.2755 | 9.0000 | 1.0000 |
| coarse_to_fine_query_bank | 0.9062 | 0.1129 | 1.3315 | 0.4612 | 7.0000 | 2.0000 |
