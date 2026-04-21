# Toy Sensitivity Budget Bridge

Matched-budget comparison across allocation policies.

| Scenario | Method | Accuracy | Uniform acc. | Δ acc. | Help vs uniform | Harm vs uniform | Protected fraction | Allocation entropy | Outlier mass | Bytes estimate | Selected slots | Selected channels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| aligned | uniform_allocation | 0.9167 | 0.9167 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 2.8315 | 0.1100 | 58763.0000 | [0, 4, 8] | [0, 6, 12, 18] |
| aligned | sensitivity_protected | 0.9688 | 0.9167 | 0.0521 | 0.0677 | 0.0156 | 0.3750 | 1.6711 | 0.1100 | 58763.0000 | [1, 4, 7] | [1, 2, 7, 12] |
| aligned | oracle_allocation | 0.9688 | 0.9167 | 0.0521 | 0.0677 | 0.0156 | 0.3750 | 2.1257 | 0.1100 | 58763.0000 | [1, 4, 7] | [1, 2, 7, 12] |
| rotated | uniform_allocation | 0.9323 | 0.9323 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 2.8315 | 0.1083 | 58763.0000 | [0, 4, 8] | [0, 6, 12, 18] |
| rotated | sensitivity_protected | 0.9583 | 0.9323 | 0.0260 | 0.0521 | 0.0260 | 0.3750 | 2.0987 | 0.1083 | 58763.0000 | [1, 4, 7] | [14, 19, 20, 21] |
| rotated | oracle_allocation | 0.9635 | 0.9323 | 0.0312 | 0.0469 | 0.0156 | 0.3750 | 2.3910 | 0.1083 | 58763.0000 | [1, 4, 7] | [13, 19, 20, 22] |
| outlier | uniform_allocation | 0.9062 | 0.9062 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 2.8315 | 0.6560 | 58763.0000 | [0, 4, 8] | [0, 6, 12, 18] |
| outlier | sensitivity_protected | 0.9531 | 0.9062 | 0.0469 | 0.0677 | 0.0208 | 0.3750 | 0.2981 | 0.6560 | 58763.0000 | [1, 2, 7] | [1, 3, 4, 11] |
| outlier | oracle_allocation | 0.9896 | 0.9062 | 0.0833 | 0.0833 | 0.0000 | 0.3750 | 1.7806 | 0.6560 | 58763.0000 | [1, 2, 7] | [1, 4, 7, 11] |
| slot_permuted | uniform_allocation | 0.9167 | 0.9167 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 2.8315 | 0.1095 | 58763.0000 | [0, 4, 8] | [0, 6, 12, 18] |
| slot_permuted | sensitivity_protected | 0.9271 | 0.9167 | 0.0104 | 0.0625 | 0.0521 | 0.3750 | 1.7964 | 0.1095 | 58763.0000 | [1, 7, 8] | [1, 2, 7, 17] |
| slot_permuted | oracle_allocation | 0.9271 | 0.9167 | 0.0104 | 0.0625 | 0.0521 | 0.3750 | 2.1257 | 0.1095 | 58763.0000 | [1, 7, 8] | [1, 2, 7, 12] |

## Scenario Averages

| Scenario | Best method | Best accuracy | Uniform acc. | Mean accuracy | Mean help vs uniform | Mean harm vs uniform |
|---|---|---:|---:|---:|---:|---:|
| aligned | sensitivity_protected | 0.9688 | 0.9167 | 0.9514 | 0.0451 | 0.0104 |
| outlier | oracle_allocation | 0.9896 | 0.9062 | 0.9497 | 0.0503 | 0.0069 |
| rotated | oracle_allocation | 0.9635 | 0.9323 | 0.9514 | 0.0330 | 0.0139 |
| slot_permuted | sensitivity_protected | 0.9271 | 0.9167 | 0.9236 | 0.0417 | 0.0347 |
