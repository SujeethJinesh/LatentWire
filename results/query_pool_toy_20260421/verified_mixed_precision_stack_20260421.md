# Toy Verified Mixed-Precision Stack

- Protected rate = fraction of atoms assigned high precision or left unpruned in a prune-only policy.
- Top-atom preservation = overlap with the oracle selection for the relevant decision stage.

| Method | Accuracy | MSE | Prune rate | Kept rate | Protected rate | Missed help | False prune | Top-atom preservation | Protected-top preservation | Bytes proxy | Compute proxy | Help vs full | Harm vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_precision | 0.9219 | 0.2488 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 772.0000 | 768.0000 | 0.0000 | 0.0000 |
| uniform_low_bit | 0.9271 | 0.2586 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 148.0000 | 768.0000 | 0.0052 | 0.0000 |
| activation_aware_quant_only | 0.9219 | 0.2594 | 0.0000 | 1.0000 | 0.2500 | 0.0000 | 0.0000 | 1.0000 | 0.3168 | 214.0000 | 768.0000 | 0.0000 | 0.0000 |
| verifier_prune_only | 0.9219 | 0.2245 | 0.2086 | 0.7914 | 0.7914 | 0.0532 | 0.1488 | 0.8646 | 0.9878 | 611.8333 | 730.2968 | 0.0000 | 0.0000 |
| prune_then_uniform_quant | 0.9323 | 0.2380 | 0.2086 | 0.7914 | 0.0000 | 0.0532 | 0.1488 | 0.8646 | 0.0000 | 117.9688 | 730.2968 | 0.0104 | 0.0000 |
| prune_then_activation_aware_quant | 0.8958 | 0.2292 | 0.2086 | 0.7914 | 0.2500 | 0.0532 | 0.1488 | 0.8646 | 0.6693 | 183.9688 | 730.2968 | 0.0000 | 0.0260 |
| oracle_stack | 0.9375 | 0.2147 | 0.2083 | 0.7917 | 0.2500 | 0.0588 | 0.1646 | 1.0000 | 1.0000 | 184.0000 | 726.4584 | 0.0156 | 0.0000 |
