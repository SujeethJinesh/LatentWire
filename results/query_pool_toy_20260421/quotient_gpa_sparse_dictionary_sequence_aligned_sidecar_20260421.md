# Toy Quotient + GPA Sparse Dictionary Sequence-Aligned Sidecar

- seed: `0`
- held-out shot grid: `[1, 2, 4, 8]`
- interface strength: `2.5`
- remap recovery: `0.7`
- byte sidecar hash dim: `16`
- alignment hash dim: `16`
- byte sidecar scale: `0.35`
- alignment sidecar scale: `0.3`
- alignment profile scale: `0.2`

This toy keeps the current quotient+GPA+sparse-dictionary pipeline fixed and asks whether a sequence-aligned interface sidecar adds useful signal beyond the plain byte-sidecar branch under the same strong tokenizer-like corruption.

References:
- Cross-Tokenizer LLM Distillation through a Byte-Level Interface: https://arxiv.org/abs/2604.07466
- DWA-KD: https://arxiv.org/abs/2602.21669
- TokAlign: https://arxiv.org/abs/2506.03523
- The Vision Wormhole: https://arxiv.org/abs/2602.15382

| Shot | Method | Accuracy | MSE | dAcc vs token-id few-shot | dMSE vs token-id few-shot | Boundary F1 | Remap coverage | Interface noise | Byte sidecar norm | Alignment sidecar norm | Head-match acc | Atom recovery | Shared Basis |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.1059 | 0.0000 | 0.0000 | 0.8251 | 0.3499 | 0.4373 | - | - | - | - | False |
| 1.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.1038 | 0.0000 | -0.0021 | 0.8251 | 0.3499 | 0.3286 | - | - | - | - | False |
| 1.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0587 | 0.0000 | -0.0473 | 0.8251 | 0.3499 | 0.4373 | - | - | 1.0000 | 0.1797 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0584 | 0.0000 | -0.0475 | 0.8251 | 0.3499 | 0.3286 | - | - | 1.0000 | 0.1797 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0384 | 0.0000 | -0.0675 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0384 | 0.0000 | -0.0675 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0362 | 0.0000 | -0.0698 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0360 | 0.0000 | -0.0699 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0593 | 0.0000 | -0.0466 | 0.8251 | 0.3499 | 0.0000 | - | - | 1.0000 | 0.3891 | True |
| 2.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0652 | 0.0000 | 0.0000 | 0.8251 | 0.3499 | 0.4373 | - | - | - | - | False |
| 2.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0614 | 0.0000 | -0.0038 | 0.8251 | 0.3499 | 0.3286 | - | - | - | - | False |
| 2.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0583 | 0.0000 | -0.0069 | 0.8251 | 0.3499 | 0.4373 | - | - | 1.0000 | 0.1797 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0599 | 0.0000 | -0.0053 | 0.8251 | 0.3499 | 0.3286 | - | - | 1.0000 | 0.1797 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0384 | 0.0000 | -0.0268 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0389 | 0.0000 | -0.0263 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0362 | 0.0000 | -0.0290 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0365 | 0.0000 | -0.0287 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0589 | 0.0000 | -0.0063 | 0.8251 | 0.3499 | 0.0000 | - | - | 1.0000 | 0.3891 | True |
| 4.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0228 | 0.0000 | 0.0000 | 0.8251 | 0.3499 | 0.4373 | - | - | - | - | False |
| 4.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0169 | 0.0000 | -0.0059 | 0.8251 | 0.3499 | 0.3286 | - | - | - | - | False |
| 4.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0562 | 0.0000 | 0.0334 | 0.8251 | 0.3499 | 0.4373 | - | - | 1.0000 | 0.1797 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0593 | 0.0000 | 0.0365 | 0.8251 | 0.3499 | 0.3286 | - | - | 1.0000 | 0.1797 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0380 | 0.0000 | 0.0152 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0386 | 0.0000 | 0.0158 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0358 | 0.0000 | 0.0130 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0362 | 0.0000 | 0.0134 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0583 | 0.0000 | 0.0355 | 0.8251 | 0.3499 | 0.0000 | - | - | 1.0000 | 0.3891 | True |
| 8.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0070 | 0.0000 | 0.0000 | 0.8251 | 0.3499 | 0.4373 | - | - | - | - | False |
| 8.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0035 | 0.0000 | -0.0035 | 0.8251 | 0.3499 | 0.3286 | - | - | - | - | False |
| 8.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0578 | 0.0000 | 0.0508 | 0.8251 | 0.3499 | 0.4373 | - | - | 1.0000 | 0.1797 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0587 | 0.0000 | 0.0517 | 0.8251 | 0.3499 | 0.3286 | - | - | 1.0000 | 0.1797 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0382 | 0.0000 | 0.0312 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0383 | 0.0000 | 0.0313 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | - | 1.0000 | 0.1797 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0359 | 0.0000 | 0.0289 | 0.8251 | 0.3499 | 0.4373 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0359 | 0.0000 | 0.0289 | 0.8251 | 0.3499 | 0.3286 | 1.0000 | 0.3606 | 1.0000 | 0.1797 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0585 | 0.0000 | 0.0515 | 0.8251 | 0.3499 | 0.0000 | - | - | 1.0000 | 0.3891 | True |

## Best by Shot
- 1 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` (0.0360); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` (0.0360)
- 2 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id` (0.0362); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id` (0.0362)
- 4 shot/class: lowest MSE `heldout_fewshot_ridge_byte_span_remap` (0.0169); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id` (0.0358)
- 8 shot/class: lowest MSE `heldout_fewshot_ridge_byte_span_remap` (0.0035); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` (0.0359)
