# Toy Quotient + GPA Sparse Dictionary Sequence-Aligned Sidecar

- seed: `1`
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
| 1.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0988 | 0.0000 | 0.0000 | 0.8255 | 0.3490 | 0.4362 | - | - | - | - | False |
| 1.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0943 | 0.0000 | -0.0045 | 0.8255 | 0.3490 | 0.3281 | - | - | - | - | False |
| 1.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0582 | 0.0000 | -0.0406 | 0.8255 | 0.3490 | 0.4362 | - | - | 1.0000 | 0.2000 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0598 | 0.0000 | -0.0391 | 0.8255 | 0.3490 | 0.3281 | - | - | 1.0000 | 0.1172 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0389 | 0.0000 | -0.0600 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | - | 1.0000 | 0.2000 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0391 | 0.0000 | -0.0597 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | - | 1.0000 | 0.1172 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0362 | 0.0000 | -0.0626 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | 0.3606 | 1.0000 | 0.2000 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0363 | 0.0000 | -0.0625 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | 0.3606 | 1.0000 | 0.1172 | True |
| 1.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0567 | 0.0000 | -0.0422 | 0.8255 | 0.3490 | 0.0000 | - | - | 1.0000 | 0.1203 | True |
| 2.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0501 | 0.0000 | 0.0000 | 0.8255 | 0.3490 | 0.4362 | - | - | - | - | False |
| 2.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0476 | 0.0000 | -0.0025 | 0.8255 | 0.3490 | 0.3281 | - | - | - | - | False |
| 2.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0587 | 0.0000 | 0.0086 | 0.8255 | 0.3490 | 0.4362 | - | - | 1.0000 | 0.2000 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0574 | 0.0000 | 0.0073 | 0.8255 | 0.3490 | 0.3281 | - | - | 1.0000 | 0.1172 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0388 | 0.0000 | -0.0113 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | - | 1.0000 | 0.2000 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0388 | 0.0000 | -0.0113 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | - | 1.0000 | 0.1172 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0362 | 0.0000 | -0.0139 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | 0.3606 | 1.0000 | 0.2000 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0361 | 0.0000 | -0.0140 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | 0.3606 | 1.0000 | 0.1172 | True |
| 2.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0575 | 0.0000 | 0.0074 | 0.8255 | 0.3490 | 0.0000 | - | - | 1.0000 | 0.1203 | True |
| 4.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0253 | 0.0000 | 0.0000 | 0.8255 | 0.3490 | 0.4362 | - | - | - | - | False |
| 4.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0206 | 0.0000 | -0.0047 | 0.8255 | 0.3490 | 0.3281 | - | - | - | - | False |
| 4.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0575 | 0.0000 | 0.0322 | 0.8255 | 0.3490 | 0.4362 | - | - | 1.0000 | 0.2000 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0578 | 0.0000 | 0.0325 | 0.8255 | 0.3490 | 0.3281 | - | - | 1.0000 | 0.1172 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0386 | 0.0000 | 0.0133 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | - | 1.0000 | 0.2000 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0388 | 0.0000 | 0.0135 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | - | 1.0000 | 0.1172 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0362 | 0.0000 | 0.0109 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | 0.3606 | 1.0000 | 0.2000 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0361 | 0.0000 | 0.0108 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | 0.3606 | 1.0000 | 0.1172 | True |
| 4.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0559 | 0.0000 | 0.0306 | 0.8255 | 0.3490 | 0.0000 | - | - | 1.0000 | 0.1203 | True |
| 8.0000 | heldout_fewshot_ridge_token_id | 1.0000 | 0.0076 | 0.0000 | 0.0000 | 0.8255 | 0.3490 | 0.4362 | - | - | - | - | False |
| 8.0000 | heldout_fewshot_ridge_byte_span_remap | 1.0000 | 0.0044 | 0.0000 | -0.0033 | 0.8255 | 0.3490 | 0.3281 | - | - | - | - | False |
| 8.0000 | quotient_gpa_sparse_dictionary_token_id | 1.0000 | 0.0569 | 0.0000 | 0.0492 | 0.8255 | 0.3490 | 0.4362 | - | - | 1.0000 | 0.2000 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_byte_span_remap | 1.0000 | 0.0566 | 0.0000 | 0.0489 | 0.8255 | 0.3490 | 0.3281 | - | - | 1.0000 | 0.1172 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_token_id | 1.0000 | 0.0386 | 0.0000 | 0.0310 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | - | 1.0000 | 0.2000 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_byte_sidecar_remap | 1.0000 | 0.0387 | 0.0000 | 0.0310 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | - | 1.0000 | 0.1172 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id | 1.0000 | 0.0362 | 0.0000 | 0.0285 | 0.8255 | 0.3490 | 0.4362 | 1.0000 | 0.3606 | 1.0000 | 0.2000 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap | 1.0000 | 0.0361 | 0.0000 | 0.0284 | 0.8255 | 0.3490 | 0.3281 | 1.0000 | 0.3606 | 1.0000 | 0.1172 | True |
| 8.0000 | quotient_gpa_sparse_dictionary_oracle_interface | 1.0000 | 0.0567 | 0.0000 | 0.0491 | 0.8255 | 0.3490 | 0.0000 | - | - | 1.0000 | 0.1203 | True |

## Best by Shot
- 1 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id` (0.0362); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id` (0.0362)
- 2 shot/class: lowest MSE `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` (0.0361); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` (0.0361)
- 4 shot/class: lowest MSE `heldout_fewshot_ridge_byte_span_remap` (0.0206); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` (0.0361)
- 8 shot/class: lowest MSE `heldout_fewshot_ridge_byte_span_remap` (0.0044); best shared-basis `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` (0.0361)
