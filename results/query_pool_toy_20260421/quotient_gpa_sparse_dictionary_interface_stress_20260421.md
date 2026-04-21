# Toy Quotient + GPA Sparse Dictionary Interface Stress

- seed: `7`
- held-out shot grid: `[1, 2, 4]`
- remap capacity: `10`
- interface strength: `2.5`
- remap recovery: `0.7`

This toy stresses the current best low-shot shared-basis lane under strong tokenizer-like interface corruption. It compares raw token-id interface transfer against a learned shared byte/span remap and an oracle interface while keeping the quotient+GPA+sparse-dictionary pipeline fixed.

References:
- TokAlign: https://arxiv.org/abs/2506.03523
- Byte Latent Transformer: https://arxiv.org/abs/2412.09871
- Complete Characterization of Gauge Symmetries in Transformer Architectures: https://openreview.net/forum?id=KrkbYbK0cH

| Shot | Method | Accuracy | MSE | dAcc vs token-id few-shot | dMSE vs token-id few-shot | Boundary F1 | Remap coverage | Interface noise | Head-match acc | Atom recovery | Shared Basis |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0989 | 0.0000 | 0.0000 | 0.8254 | 0.3492 | 0.4365 | - | - | False |
| 1.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.1005 | 0.0000 | 0.0016 | 0.8254 | 0.3492 | 0.3282 | - | - | False |
| 1.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0582 | 0.0000 | -0.0408 | 0.8254 | 0.3492 | 0.4365 | 1.0000 | 0.1975 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0566 | 0.0000 | -0.0423 | 0.8254 | 0.3492 | 0.3282 | 1.0000 | 0.1950 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0568 | 0.0000 | -0.0421 | 0.8254 | 0.3492 | 0.0000 | 1.0000 | 0.1950 | True |
| 2.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0617 | 0.0000 | 0.0000 | 0.8254 | 0.3492 | 0.4365 | - | - | False |
| 2.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0600 | 0.0000 | -0.0017 | 0.8254 | 0.3492 | 0.3282 | - | - | False |
| 2.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0579 | 0.0000 | -0.0038 | 0.8254 | 0.3492 | 0.4365 | 1.0000 | 0.1975 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0570 | 0.0000 | -0.0046 | 0.8254 | 0.3492 | 0.3282 | 1.0000 | 0.1950 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0576 | 0.0000 | -0.0040 | 0.8254 | 0.3492 | 0.0000 | 1.0000 | 0.1950 | True |
| 4.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0326 | 0.0000 | 0.0000 | 0.8254 | 0.3492 | 0.4365 | - | - | False |
| 4.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0238 | 0.0000 | -0.0087 | 0.8254 | 0.3492 | 0.3282 | - | - | False |
| 4.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0567 | 0.0000 | 0.0241 | 0.8254 | 0.3492 | 0.4365 | 1.0000 | 0.1975 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0568 | 0.0000 | 0.0243 | 0.8254 | 0.3492 | 0.3282 | 1.0000 | 0.1950 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0555 | 0.0000 | 0.0229 | 0.8254 | 0.3492 | 0.0000 | 1.0000 | 0.1950 | True |

## Best by Shot
- 1 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_byte_span_remap` (0.0566); best shared-basis `quotient_gpa_sparse_dictionary_byte_span_remap` (0.0566)
- 2 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_byte_span_remap` (0.0570); best shared-basis `quotient_gpa_sparse_dictionary_byte_span_remap` (0.0570)
- 4 shot/class: lowest MSE `heldout_fewshot_ridge_byte_span_remap` (0.0238); best shared-basis `quotient_gpa_sparse_dictionary_oracle_interface` (0.0555)
