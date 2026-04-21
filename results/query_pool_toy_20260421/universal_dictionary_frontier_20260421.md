# Toy Universal Dictionary Frontier

A deterministic protected-frontier toy for SAE/universal-dictionary-inspired selection.

| Method | Accuracy | Acc delta | MSE | MSE delta | Feature persistence | Patch-rank corr | Selector stability | Protected-oracle preservation | Bytes proxy | Compute proxy | Help vs prune-uniform | Harm vs prune-uniform |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prune_uniform_quant | 0.7083 | 0.0000 | 0.0914 | 0.0000 | 0.0000 | -0.0117 | 0.0000 | 0.0000 | 154.0000 | 420.0000 | 0.0000 | 0.0000 |
| raw_activation_protect | 1.0000 | 0.2917 | 0.0382 | -0.0532 | 0.0247 | 0.8468 | 1.0000 | 0.5000 | 250.0000 | 446.4000 | 0.2917 | 0.0000 |
| quant_error_protect | 1.0000 | 0.2917 | 0.0324 | -0.0591 | 0.0125 | 0.8675 | 1.0000 | 0.5000 | 250.0000 | 446.4000 | 0.2917 | 0.0000 |
| exact_patch_effect_protect | 1.0000 | 0.2917 | 0.0318 | -0.0596 | 0.0394 | 1.0000 | 1.0000 | 1.0000 | 250.0000 | 446.4000 | 0.2917 | 0.0000 |
| universal_dictionary_persistence_protect | 1.0000 | 0.2917 | 0.0393 | -0.0522 | 0.3303 | 0.6078 | 1.0000 | 0.5000 | 250.0000 | 446.4000 | 0.2917 | 0.0000 |
| random_protect | 0.7083 | 0.0000 | 0.0483 | -0.0431 | 0.4906 | 0.0429 | 0.0909 | 0.3333 | 250.0000 | 446.4000 | 0.0000 | 0.0000 |
| utility_oracle_protect | 0.7188 | 0.0104 | 0.0809 | -0.0105 | 0.9577 | -0.5753 | 1.0000 | 0.0000 | 250.0000 | 446.4000 | 0.0104 | 0.0000 |

Interpretation: a useful shared dictionary selector should be more stable than random, preserve more exact patch-effect atoms than raw activation, and improve MSE without relying on a utility oracle.
