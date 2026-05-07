# real_hbsm_b1_sensitivity_packet

Decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`.

Rows: `56`.

Aggregate gate fields:
- `baseline_spearman`: `{'layer_index': 0.47619047619047616, 'parameter_count_norm': -0.5773502691896257, 'weight_norm': -0.47619047619047616, 'boundary_flag': 0.1690308509457033, 'kl_lens_rank': 1.0, 'activation_outlier': 0.09523809523809523}`
- `boundary_random_top_decile_count`: `0`
- `boundary_random_top_decile_rate`: `0.0`
- `boundary_top_decile_count`: `1`
- `boundary_top_decile_enrichment`: `333333333.3333333`
- `boundary_top_decile_rate`: `0.3333333333333333`
- `cheap_predictor_margin_vs_best_baseline`: `-1.4761904761904763`
- `cheap_predictor_spearman`: `-0.47619047619047616`
- `control_types`: `['activation_outlier', 'boundary_only', 'kl_lens_rank', 'layer_index', 'parameter_count_norm', 'perturbation_off', 'random_flags']`
- `expected_top_decile_count`: `1`
- `fisher_p_boundary_top_decile`: `0.375`
- `fisher_p_random_boundary_top_decile`: `1.0`
- `gate_name`: `hbsm_b1_boundary_sensitivity_enrichment`
- `gate_pass`: `False`
- `gate_status`: `FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`
- `non_boundary_random_top_decile_count`: `1`
- `non_boundary_random_top_decile_rate`: `0.2`
- `non_boundary_top_decile_count`: `0`
- `non_boundary_top_decile_rate`: `0.0`
- `primary_row_count`: `8`
- `prompt_count`: `1`
- `random_boundary_top_decile_enrichment`: `0.0`
- `random_top_decile_count`: `1`
- `scoring_layer_count`: `8`
- `split_counts`: `{'train': 4, 'test': 4}`
- `test_count`: `4`
- `top_decile_count`: `1`
- `train_count`: `4`

This packet contains saved-tensor measurements only. It is not GPU throughput, HBM, or latency evidence.
