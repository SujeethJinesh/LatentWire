# Toy Quantization Bridge

| Method | Rotation | Quantizer | MSE | Cosine | Outlier energy retained | Bytes estimate |
|---|---|---|---:|---:|---:|---:|
| none_uniform | none | uniform | 0.2646 | 0.9868 | 1.0068 | 20.0000 |
| none_protected_outlier | none | protected_outlier | 0.0077 | 0.9994 | 1.0000 | 30.0000 |
| random_uniform | random | uniform | 0.0870 | 0.9954 | 1.0023 | 20.0000 |
| random_protected_outlier | random | protected_outlier | 0.0550 | 0.9970 | 1.0019 | 30.0000 |
| hadamard_uniform | hadamard | uniform | 0.0575 | 0.9968 | 1.0010 | 20.0000 |
| hadamard_protected_outlier | hadamard | protected_outlier | 0.0491 | 0.9973 | 0.9999 | 30.0000 |

Interpretation:

The toy matches the quantization literature's expected tradeoff. If the
outlier basis is known, protected-channel quantization sharply reduces MSE
(`0.2646 -> 0.0077`) at a larger byte estimate (`20 -> 30`). If we first rotate
away the coordinate outliers, uniform quantization becomes much stronger
(`0.2646 -> 0.0575` under Hadamard), and the protected escape path adds only a
smaller improvement. For LatentWire, this supports a concrete ablation:
separate basis fixing from outlier protection before claiming either is the
source of a bridge gain.
