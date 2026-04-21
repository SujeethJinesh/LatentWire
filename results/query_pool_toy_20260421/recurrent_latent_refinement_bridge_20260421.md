# Toy Recurrent Latent Refinement Bridge

| Method | Accuracy | MSE | Cosine | Steps | Bytes estimate | Compute proxy | Help rate | Harm rate | Trajectory MSE start | Trajectory MSE end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one_shot_bridge | 0.6458 | 0.0508 | 0.7502 | 1.0000 | 16.0000 | 31.2000 | 0.0000 | 0.0000 | 0.0508 | 0.0508 |
| two_step_residual_refinement | 0.5000 | 0.0315 | 0.7605 | 2.0000 | 27.0000 | 39.6000 | 0.8958 | 0.1042 | 0.0354 | 0.0315 |
| gated_refinement | 0.6562 | 0.0203 | 0.8569 | 2.0000 | 19.0000 | 39.6000 | 1.0000 | 0.0000 | 0.0199 | 0.0203 |
| blockwise_diffusion_denoise | 0.7083 | 0.0170 | 0.8741 | 5.0000 | 20.0000 | 64.8000 | 1.0000 | 0.0000 | 0.0469 | 0.0170 |
| oracle_upper_bound | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
