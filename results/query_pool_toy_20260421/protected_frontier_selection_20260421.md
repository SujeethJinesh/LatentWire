# Toy Protected Frontier Selection

- All methods share the same verifier-pruned frontier; only the high-precision protected subset changes.
- Protected-oracle preservation measures overlap with atoms whose low-bit quantization error most damages positive utility.

| Method | Accuracy | Acc delta | MSE | MSE delta | Prune rate | Protected rate | Missed help | False prune | Top-atom preservation | Protected-oracle preservation | Protection precision | Bytes proxy | Compute proxy | Help vs prune-uniform | Harm vs prune-uniform |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prune_uniform_quant | 0.6615 | 0.0000 | 0.2543 | 0.0000 | 0.3244 | 0.0000 | 0.0111 | 0.0092 | 0.7859 | 0.0000 | 0.0000 | 89.5833 | 681.0000 | 0.0000 | 0.0000 |
| global_activation_protect | 0.7917 | 0.1302 | 0.1905 | -0.0638 | 0.3244 | 0.2143 | 0.0111 | 0.0092 | 0.7859 | 0.5382 | 0.5547 | 176.5833 | 681.0000 | 0.1302 | 0.0000 |
| verifier_saliency_protect | 0.7708 | 0.1094 | 0.2229 | -0.0315 | 0.3244 | 0.2143 | 0.0111 | 0.0092 | 0.7859 | 0.4089 | 0.5330 | 176.5833 | 681.0000 | 0.1250 | 0.0156 |
| quant_error_protect | 0.8073 | 0.1458 | 0.1861 | -0.0682 | 0.3244 | 0.2143 | 0.0111 | 0.0092 | 0.7859 | 0.6476 | 0.5538 | 176.5833 | 681.0000 | 0.1458 | 0.0000 |
| activation_x_verifier_protect | 0.7812 | 0.1198 | 0.2127 | -0.0416 | 0.3244 | 0.2143 | 0.0111 | 0.0092 | 0.7859 | 0.4826 | 0.5547 | 176.5833 | 681.0000 | 0.1250 | 0.0052 |
| random_protect | 0.7135 | 0.0521 | 0.2352 | -0.0191 | 0.3244 | 0.2143 | 0.0111 | 0.0092 | 0.7859 | 0.3003 | 0.4080 | 176.5833 | 681.0000 | 0.0729 | 0.0208 |
| oracle_protect | 0.8073 | 0.1458 | 0.1698 | -0.0845 | 0.3244 | 0.2143 | 0.0111 | 0.0092 | 0.7859 | 1.0000 | 0.5521 | 176.5833 | 681.0000 | 0.1458 | 0.0000 |
