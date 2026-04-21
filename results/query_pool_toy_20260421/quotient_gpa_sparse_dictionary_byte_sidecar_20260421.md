# Toy Quotient + GPA Sparse Dictionary Byte Sidecar

- seed: `7`
- held-out shot grid: `[1, 2, 4]`
- interface strength: `2.5`
- remap recovery: `0.7`
- sidecar hash dim: `16`
- sidecar ngram max: `2`
- sidecar scale: `0.35`

This toy asks whether a tokenizer-agnostic byte sidecar adds useful interface information on top of the current best low-shot shared-basis lane. It keeps the quotient+GPA+sparse-dictionary pipeline fixed and compares token-id transfer, byte/span remap, and a hashed byte-sidecar augmentation under strong interface corruption.

References:
- Cross-Tokenizer LLM Distillation through a Byte-Level Interface: https://arxiv.org/abs/2604.07466
- DWA-KD: https://arxiv.org/abs/2602.21669
- The Vision Wormhole: https://arxiv.org/abs/2602.15382
- Latent-DARM: https://arxiv.org/abs/2603.09184

| Shot | Method | Accuracy | MSE | dAcc vs token-id few-shot | dMSE vs token-id few-shot | Boundary F1 | Remap coverage | Interface noise | Sidecar norm | Head-match acc | Atom recovery | Shared Basis |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0989 | 0.0000 | 0.0000 | 0.8254 | 0.3492 | 0.4365 | - | - | - | False |
| 1.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.1005 | 0.0000 | 0.0016 | 0.8254 | 0.3492 | 0.3282 | - | - | - | False |
| 1.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0582 | 0.0000 | -0.0408 | 0.8254 | 0.3492 | 0.4365 | - | 1.0000 | 0.1975 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0566 | 0.0000 | -0.0423 | 0.8254 | 0.3492 | 0.3282 | - | 1.0000 | 0.1950 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0399 | 0.0000 | -0.0590 | 0.8254 | 0.3492 | 0.4365 | 1.0000 | 1.0000 | 0.1975 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0392 | 0.0000 | -0.0598 | 0.8254 | 0.3492 | 0.3282 | 1.0000 | 1.0000 | 0.1950 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0568 | 0.0000 | -0.0421 | 0.8254 | 0.3492 | 0.0000 | - | 1.0000 | 0.1950 | True |
| 2.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0617 | 0.0000 | 0.0000 | 0.8254 | 0.3492 | 0.4365 | - | - | - | False |
| 2.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0600 | 0.0000 | -0.0017 | 0.8254 | 0.3492 | 0.3282 | - | - | - | False |
| 2.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0579 | 0.0000 | -0.0038 | 0.8254 | 0.3492 | 0.4365 | - | 1.0000 | 0.1975 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0570 | 0.0000 | -0.0046 | 0.8254 | 0.3492 | 0.3282 | - | 1.0000 | 0.1950 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0395 | 0.0000 | -0.0222 | 0.8254 | 0.3492 | 0.4365 | 1.0000 | 1.0000 | 0.1975 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0394 | 0.0000 | -0.0222 | 0.8254 | 0.3492 | 0.3282 | 1.0000 | 1.0000 | 0.1950 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0576 | 0.0000 | -0.0040 | 0.8254 | 0.3492 | 0.0000 | - | 1.0000 | 0.1950 | True |
| 4.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0326 | 0.0000 | 0.0000 | 0.8254 | 0.3492 | 0.4365 | - | - | - | False |
| 4.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0238 | 0.0000 | -0.0087 | 0.8254 | 0.3492 | 0.3282 | - | - | - | False |
| 4.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0567 | 0.0000 | 0.0241 | 0.8254 | 0.3492 | 0.4365 | - | 1.0000 | 0.1975 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0568 | 0.0000 | 0.0243 | 0.8254 | 0.3492 | 0.3282 | - | 1.0000 | 0.1950 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0390 | 0.0000 | 0.0064 | 0.8254 | 0.3492 | 0.4365 | 1.0000 | 1.0000 | 0.1975 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0393 | 0.0000 | 0.0067 | 0.8254 | 0.3492 | 0.3282 | 1.0000 | 1.0000 | 0.1950 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0555 | 0.0000 | 0.0229 | 0.8254 | 0.3492 | 0.0000 | - | 1.0000 | 0.1950 | True |

## Best by Shot
- 1 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_byte_sidecar_remap` (0.0392); best shared-basis `quotient_gpa_sparse_dictionary_byte_sidecar_remap` (0.0392)
- 2 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_byte_sidecar_remap` (0.0394); best shared-basis `quotient_gpa_sparse_dictionary_byte_sidecar_remap` (0.0394)
- 4 shot/class: lowest MSE `heldout_fewshot_ridge_byte_span_remap` (0.0238); best shared-basis `quotient_gpa_sparse_dictionary_byte_sidecar_token_id` (0.0390)
