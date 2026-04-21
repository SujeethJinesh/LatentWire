# Toy Activation-Aware Atom Quant

| Method | Acc | MSE | Cosine | Bit budget | Bytes proxy | Protected rate | Outlier protected | Top-atom preservation | Help vs full | Harm vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_precision | 1.0000 | 0.0000 | 1.0000 | 16.0000 | 68.0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| uniform_low_bit | 0.9531 | 0.7985 | 0.9705 | 3.0000 | 16.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0469 |
| random_mixed_precision | 0.9792 | 0.4235 | 0.9837 | 4.2500 | 29.0 | 0.2500 | 0.2500 | 0.5000 | 0.0000 | 0.0208 |
| activation_aware_mixed_precision | 1.0000 | 0.0256 | 0.9988 | 4.2500 | 29.0 | 0.2500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| protected_outlier_mixed_precision | 1.0000 | 0.0256 | 0.9988 | 4.2500 | 29.0 | 0.2500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| oracle_mixed_precision | 1.0000 | 0.0256 | 0.9988 | 4.2500 | 29.0 | 0.2500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
