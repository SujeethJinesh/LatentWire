# Public Adapter Held-Out Packet Summary

- pass gate: `True`
- rows: `36`
- pass rows: `15`
- near-miss rows: `15`
- semantic-anchor reference pass rows: `18/18`
- semantic-anchor min passing accuracy: `0.75`
- semantic-anchor min CI95 low: `0.45703125`

## Runs

| Run | Model | Mode | Adapter target | Pass rows | Direction pass | Max acc | Max delta | Max CI95 low | Min oracle |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n256_minilm_teacher_norm_dec048_evaldisjoint` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean/candidate_local_residual_norm` | `semantic_anchor_teacher` | 4 | `{'core_to_holdout': True, 'holdout_to_core': True, 'same_family_all': True}` | 0.625 | 0.375 | 0.320 | 0.625 |
| `results/source_private_candidate_local_residual_receiver_20260430_seed53_n256_minilm_teacher_norm_dec048_evaldisjoint` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean/candidate_local_residual_norm` | `semantic_anchor_teacher` | 4 | `{'core_to_holdout': True, 'holdout_to_core': True, 'same_family_all': True}` | 0.625 | 0.375 | 0.320 | 0.625 |
| `results/source_private_candidate_local_residual_receiver_20260430_seed59_n256_minilm_teacher_norm_dec048_evaldisjoint` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean/candidate_local_residual_norm` | `semantic_anchor_teacher` | 4 | `{'core_to_holdout': True, 'holdout_to_core': True, 'same_family_all': True}` | 0.625 | 0.375 | 0.316 | 0.625 |
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n512_minilm_teacher_norm_dec048_evaldisjoint` | sentence-transformers/all-MiniLM-L6-v2 | `hf_last_mean/candidate_local_residual_norm` | `semantic_anchor_teacher` | 3 | `{'core_to_holdout': True, 'holdout_to_core': True, 'same_family_all': True}` | 0.625 | 0.375 | 0.332 | 0.625 |

## Best Rows

| Run | Direction | Budget | Pass | Acc | Target | Best ctrl | Delta | CI95 low | Oracle | Top knock | Random knock | Controls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n512_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 8 | `True` | 0.625 | 0.250 | 0.250 | 0.375 | 0.332 | 0.875 | 1.000 | 0.078 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n256_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 8 | `True` | 0.625 | 0.250 | 0.250 | 0.375 | 0.316 | 0.875 | 1.000 | 0.073 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed59_n256_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 8 | `True` | 0.625 | 0.250 | 0.250 | 0.375 | 0.316 | 0.875 | 1.000 | 0.062 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed53_n256_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 8 | `True` | 0.625 | 0.250 | 0.258 | 0.375 | 0.320 | 0.875 | 1.000 | 0.000 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n512_minilm_teacher_norm_dec048_evaldisjoint` | same_family_all | 8 | `True` | 0.562 | 0.250 | 0.250 | 0.312 | 0.273 | 0.875 | 1.000 | 0.487 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed59_n256_minilm_teacher_norm_dec048_evaldisjoint` | same_family_all | 8 | `True` | 0.562 | 0.250 | 0.250 | 0.312 | 0.262 | 0.875 | 1.000 | 0.525 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n256_minilm_teacher_norm_dec048_evaldisjoint` | same_family_all | 8 | `True` | 0.562 | 0.250 | 0.250 | 0.312 | 0.258 | 0.875 | 1.000 | 0.450 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed53_n256_minilm_teacher_norm_dec048_evaldisjoint` | same_family_all | 8 | `True` | 0.562 | 0.250 | 0.258 | 0.312 | 0.258 | 0.875 | 1.000 | 0.625 | `True` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n512_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 4 | `False` | 0.625 | 0.250 | 0.375 | 0.375 | 0.332 | 0.750 | 1.000 | 0.661 | `False` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed47_n256_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 4 | `False` | 0.625 | 0.250 | 0.375 | 0.375 | 0.320 | 0.750 | 1.000 | 0.562 | `False` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed53_n256_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 4 | `False` | 0.625 | 0.250 | 0.375 | 0.375 | 0.316 | 0.750 | 1.000 | 0.521 | `False` |
| `results/source_private_candidate_local_residual_receiver_20260430_seed59_n256_minilm_teacher_norm_dec048_evaldisjoint` | core_to_holdout | 4 | `False` | 0.625 | 0.250 | 0.375 | 0.375 | 0.312 | 0.750 | 1.000 | 0.490 | `False` |

## Interpretation

Candidate-local residual scoring turns the public semantic-anchor adapter into a positive bidirectional held-out packet result on this sweep. The surviving rows require matched source packets to beat target-only and all strict destructive controls, including private-random atom packets and an in-run permuted-teacher receiver. The branch should now be scaled to larger frozen slices, repeated seeds, and competitor baselines before being treated as ICLR-ready.
