# Toy Iterative Latent Refinement

Deterministic ablation for target-side latent repair after noisy cross-model transfer.

| Method | Accuracy | MSE | Reg MSE | Steps | Compute | Bytes | Help | Harm | MSE Help | MSE Harm | Confidence | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one_pass_bridge | 0.9625 | 0.0559 | 0.9821 | 1.0000 | 24.0000 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9158 | 0.0467 |
| fixed_2_step_refinement | 0.9563 | 0.0449 | 0.5822 | 2.0000 | 33.1200 | 25.0000 | 0.0250 | 0.0312 | 0.8250 | 0.1750 | 0.9147 | 0.0447 |
| fixed_4_step_refinement | 0.9125 | 0.0673 | 0.5228 | 4.0000 | 51.3600 | 43.0000 | 0.0312 | 0.0812 | 0.6562 | 0.3438 | 0.9301 | 0.0336 |
| confidence_gated_refinement | 0.9375 | 0.0619 | 0.9193 | 2.0000 | 25.1400 | 18.2500 | 0.0312 | 0.0562 | 0.0000 | 0.1250 | 0.9253 | 0.0295 |
| noisy_diffusion_refinement | 0.9500 | 0.0468 | 0.7591 | 5.0000 | 60.4800 | 40.0000 | 0.0125 | 0.0250 | 0.8500 | 0.1500 | 0.9137 | 0.0381 |
| oracle_refinement | 0.9750 | 0.0048 | 0.0648 | 2.0000 | 27.0400 | 24.0000 | 0.0250 | 0.0125 | 1.0000 | 0.0000 | 0.9185 | 0.0565 |

## Failure Reasons

- `one_pass_bridge`: low_confidence_unrepaired=6, mse_improved_but_task_wrong=0, ok_correct_improved=0, other=0, over_refined_harm=0, unchanged_correct=154, wrong_high_confidence=0
- `fixed_2_step_refinement`: low_confidence_unrepaired=1, mse_improved_but_task_wrong=0, ok_correct_improved=132, other=18, over_refined_harm=8, unchanged_correct=0, wrong_high_confidence=1
- `fixed_4_step_refinement`: low_confidence_unrepaired=0, mse_improved_but_task_wrong=0, ok_correct_improved=105, other=36, over_refined_harm=18, unchanged_correct=0, wrong_high_confidence=1
- `confidence_gated_refinement`: low_confidence_unrepaired=0, mse_improved_but_task_wrong=0, ok_correct_improved=0, other=5, over_refined_harm=14, unchanged_correct=140, wrong_high_confidence=1
- `noisy_diffusion_refinement`: low_confidence_unrepaired=3, mse_improved_but_task_wrong=3, ok_correct_improved=133, other=19, over_refined_harm=2, unchanged_correct=0, wrong_high_confidence=0
- `oracle_refinement`: low_confidence_unrepaired=0, mse_improved_but_task_wrong=4, ok_correct_improved=156, other=0, over_refined_harm=0, unchanged_correct=0, wrong_high_confidence=0

## Interpretation

Fixed target-side refinement tests whether repeated manifold repair helps after transfer; confidence gating measures selective compute; noisy diffusion adds stochastic-denoising pressure; oracle refinement upper-bounds sparse residual repair.
