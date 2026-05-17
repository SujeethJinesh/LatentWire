# Post-M18 Analysis 4: Per-Layer Mechanism Dissection
## Method
For each model, I computed strict set-leaving from prompt-averaged activation magnitudes: channels in the top-1% at position 100 that are not in the top-1% at the packet final position. This is a deterministic layer-level readout, not a trace-bootstrap confidence interval.
## Granite-Small Phase 1
Source: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`. Final position: `20000`.

| Layer type | Layer count | Mean set-leaving | Median set-leaving | Min | Max |
|---|---:|---:|---:|---:|---:|
| attention | 4 | 0.341463414634 | 0.353658536585 | 0.243902439024 | 0.414634146341 |
| mamba | 36 | 0.335365853659 | 0.329268292683 | 0.170731707317 | 0.439024390244 |

Lowest-drift layers: 1:mamba=0.170731707317, 39:mamba=0.195121951220, 31:mamba=0.243902439024, 35:attention=0.243902439024, 0:mamba=0.268292682927.
Highest-drift layers: 11:mamba=0.439024390244, 12:mamba=0.439024390244, 13:mamba=0.439024390244, 14:mamba=0.439024390244, 9:mamba=0.414634146341.

## Nemotron-3-Nano Phase 2
Source: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`. Final position: `20000`.

| Layer type | Layer count | Mean set-leaving | Median set-leaving | Min | Max |
|---|---:|---:|---:|---:|---:|
| attention | 6 | 0.358024691358 | 0.370370370370 | 0.296296296296 | 0.407407407407 |
| mamba | 23 | 0.342995169082 | 0.370370370370 | 0.148148148148 | 0.444444444444 |
| moe_expert | 23 | 0.338164251208 | 0.333333333333 | 0.148148148148 | 0.481481481481 |

Lowest-drift layers: 0:mamba=0.148148148148, 3:moe_expert=0.148148148148, 2:mamba=0.185185185185, 40:moe_expert=0.222222222222, 1:moe_expert=0.259259259259.
Highest-drift layers: 15:moe_expert=0.481481481481, 20:moe_expert=0.481481481481, 21:mamba=0.444444444444, 23:mamba=0.444444444444, 24:moe_expert=0.444444444444.

## DeepSeek-R1-Distill-Qwen-1.5B Phase 5'
Source: `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z`. Final position: `20000`.

| Layer type | Layer count | Mean set-leaving | Median set-leaving | Min | Max |
|---|---:|---:|---:|---:|---:|
| attention | 28 | 0.361607142857 | 0.375000000000 | 0.125000000000 | 0.500000000000 |

Lowest-drift layers: 0:attention=0.125000000000, 27:attention=0.187500000000, 4:attention=0.250000000000, 5:attention=0.250000000000, 1:attention=0.312500000000.
Highest-drift layers: 9:attention=0.500000000000, 3:attention=0.437500000000, 7:attention=0.437500000000, 8:attention=0.437500000000, 10:attention=0.437500000000.

## Falcon-H1 Phase 7
Source: `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z`. Final position: `20000`.

| Layer type | Layer count | Mean set-leaving | Median set-leaving | Min | Max |
|---|---:|---:|---:|---:|---:|
| post_sum_residual | 36 | 0.320707070707 | 0.318181818182 | 0.181818181818 | 0.454545454545 |

Lowest-drift layers: 28:post_sum_residual=0.181818181818, 30:post_sum_residual=0.181818181818, 31:post_sum_residual=0.181818181818, 35:post_sum_residual=0.181818181818, 0:post_sum_residual=0.272727272727.
Highest-drift layers: 13:post_sum_residual=0.454545454545, 14:post_sum_residual=0.454545454545, 18:post_sum_residual=0.454545454545, 19:post_sum_residual=0.454545454545, 22:post_sum_residual=0.454545454545.

## Interpretation
The layer-level readout does not isolate drift to one simple layer class. Granite attention and Mamba layers are close; Nemotron has measurable drift across Mamba, MoE, and attention layers; DeepSeek is all attention/MLP block residual output in this analysis; Falcon-H1 remains post-sum residual only because separate pathway hooks were not available without source modification.
