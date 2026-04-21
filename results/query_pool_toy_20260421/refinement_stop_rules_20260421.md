# Toy Refinement Stop Rules

Deterministic ablation for stopping target-side latent refinement before over-repair.

| Method | Accuracy | MSE | Reg MSE | Avg Steps | Compute | Bytes | Help | Harm | MSE Help | MSE Harm | Over-Refine | Confidence | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_1_step | 0.9625 | 0.0559 | 0.9821 | 1.0000 | 24.0000 | 16.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9158 | 0.0467 |
| fixed_2_step | 0.9563 | 0.0449 | 0.5822 | 2.0000 | 33.1200 | 25.0000 | 0.0250 | 0.0312 | 0.8250 | 0.1750 | 0.1750 | 0.9147 | 0.0447 |
| fixed_4_step | 0.9125 | 0.0673 | 0.5228 | 4.0000 | 51.3600 | 43.0000 | 0.0312 | 0.0812 | 0.6562 | 0.3438 | 0.9625 | 0.9301 | 0.0336 |
| confidence_stop | 0.9125 | 0.0693 | 0.8761 | 1.4813 | 28.3890 | 20.3313 | 0.0312 | 0.0812 | 0.0000 | 0.1813 | 0.1813 | 0.9309 | 0.0327 |
| score_drift_stop | 0.9187 | 0.0652 | 0.5187 | 3.5563 | 47.3130 | 39.0063 | 0.0312 | 0.0750 | 0.6875 | 0.3125 | 0.7937 | 0.9299 | 0.0136 |
| verifier_harm_stop | 0.9625 | 0.0614 | 0.5715 | 3.7188 | 48.7950 | 40.4688 | 0.0188 | 0.0188 | 0.6562 | 0.2500 | 0.8687 | 0.9408 | 0.0217 |
| oracle_stop | 0.9688 | 0.0406 | 0.6420 | 2.1250 | 34.2600 | 26.1250 | 0.0063 | 0.0000 | 0.8250 | 0.0000 | 0.0000 | 0.9119 | 0.0569 |

## Stop Reasons

- `fixed_1_step`: fixed_1_step=160; 1_steps=160, 2_steps=0, 3_steps=0, 4_steps=0
- `fixed_2_step`: fixed_2_step=160; 1_steps=0, 2_steps=160, 3_steps=0, 4_steps=0
- `fixed_4_step`: fixed_4_step=160; 1_steps=0, 2_steps=0, 3_steps=0, 4_steps=160
- `confidence_stop`: confidence_reached=137, max_steps=23; 1_steps=131, 2_steps=4, 3_steps=2, 4_steps=23
- `score_drift_stop`: max_steps=92, score_drift_small=68; 1_steps=0, 2_steps=3, 3_steps=65, 4_steps=92
- `verifier_harm_stop`: max_steps=145, verifier_predicted_harm=15; 1_steps=15, 2_steps=0, 3_steps=0, 4_steps=145
- `oracle_stop`: oracle_best_prefix=154, oracle_max_steps=6; 1_steps=28, 2_steps=90, 3_steps=36, 4_steps=6

## Interpretation

Stop rules isolate whether target-side repair should halt on confidence, score drift, verifier-predicted harm, or an oracle best-prefix criterion. Over-refinement rate is the main blocker metric: useful repair is not enough if later steps erase task-relevant signal.
