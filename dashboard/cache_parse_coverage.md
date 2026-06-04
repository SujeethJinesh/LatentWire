# Cache Parse Coverage

- files_seen: `8725`
- usable_files: `47`
- parsed_rows_seen: `229926`
- parsed_confirm_rows_counted_not_consumed: `46273`
- stage1_confirm_rows_consumed: `0`

| cache_family | files_seen | rows_seen | rows_dev | rows_gate | rows_confirm | parse_status | usable_for_methods | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experimental/cross_layer_error/results/cle_theoretical_20260508T191327Z | 16 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/decode_microkernel/phase0/results/decode_microkernel_phase0_20260507T233130Z | 5 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/decode_microkernel/phase1/results/dmc_phase1_20260508T000525Z | 24 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/decode_microkernel/phase2/results/dmc_phase2_20260508T0045Z | 8 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/hbsm/phase2/results/hbsm_synthetic_b1 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/horn/phase2/results/horn_synthetic_h1 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z | 12 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z | 13 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z | 15 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase3/results/layer_stratified_migration.json | 1 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z | 24 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z | 23 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase5_double_prime/results/om_phase5dp_qwen36_20260512T070500Z | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z | 14 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z | 15 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_driftrot_deepseek_clip_tight_20260528T2318Z | 25 | 12 | 9 | 1 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_driftrot_falcon_clip_tight_20260528T2343Z | 25 | 12 | 7 | 3 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z | 17 | 4 | 3 | 0 | 1 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z | 17 | 4 | 1 | 2 | 1 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z | 17 | 1 | 1 | 0 | 0 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z | 17 | 3 | 3 | 0 | 0 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_20260528T2052Z | 18 | 1 | 1 | 0 | 0 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_fixed_20260528T2116Z | 18 | 1 | 0 | 1 | 0 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z | 25 | 12 | 5 | 3 | 4 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z | 28 | 12 | 7 | 3 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z | 23 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z | 28 | 12 | 6 | 4 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z | 27 | 12 | 7 | 3 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z | 28 | 12 | 8 | 1 | 3 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z | 29 | 12 | 8 | 2 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z | 26 | 12 | 8 | 1 | 3 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_20260513T042800Z | 17 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z | 27 | 12 | 7 | 3 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_mpred_deepseek_extension_20260528T005045Z | 27 | 12 | 8 | 3 | 1 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_mpred_falcon_extension_20260528T011658Z | 27 | 12 | 7 | 2 | 3 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_mpred_granite_reduced_20260527T201251Z | 28 | 12 | 6 | 4 | 2 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_phase9_msurface_granite_internals_20260529T030958Z | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_phase9_msurface_granite_sanity_20260529T0032Z | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_stage1_e15_bca_holm_20260527T093931Z | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z | 40 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_stage1_e2_mathonly_20260526T1202Z | 112 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_stage1_e3_granite_20260526T034442Z | 26 | 12 | 9 | 3 | 0 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_stage1_e4_granite_probe_20260527T093400Z | 7 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_stage1_e5_qwen3_skipinfra_20260527T093749Z | 13 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/outlier_migrate/phase9/results/om_v1_paroquant_deepseek_20260528T162858Z | 29 | 12 | 11 | 1 | 0 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/outlier_migrate/phase9/results/om_v1_paroquant_falcon_20260528T154653Z | 30 | 12 | 6 | 3 | 3 | quarantined_unimplemented_parser,row_parsed | true |  |
| experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z | 13 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z | 15 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hbsm_local_sensitivity_20260507 | 9 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hbsm_prompt2_sensitivity_20260507 | 9 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/horn_h2_noise_replay_scout_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_architecture_maps_20260506 | 8 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_capture_manifests_20260507 | 20 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_capture_manifests_s1b_holdout_20260507 | 20 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_local_capture_preflight_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_manifest_local_capture_20260507 | 26 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_model_eligibility_20260506 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_trace_plan_20260507 | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_trace_plan_s1b_holdout_20260507 | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/hybrid_transformers_smoke_probe_20260507 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_all_layer_scout_20260507 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_local_bucket_capture_20260507 | 9 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_local_multilayer_capture_20260507 | 11 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_prompt_repeat_scout_20260507 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507 | 12 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s1b_holdout_metrics_scout_20260507 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s1b_holdout_tensor_capture_20260507 | 12 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_layer0_block256_20260507 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_mixed_block256_12p_20260507 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_block256_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_debug_20260507 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_int3_block256_12p_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_int3_block256_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_int3_block64_12p_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layer0_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layer12_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layer30_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layers0_30_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507 | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507 | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layers0_30_20260507 | 5 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_prefilter_granite_tiny_layers0_30_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer30_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507 | 6 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_transfer_prefilter_layers0_30_20260507 | 5 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_transfer_prefilter_mixed25_layers0_30_20260507 | 10 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/shared/results/ssq_lr_s3_transfer_smoke_granite_350m_layers0_30_20260507 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/sinkkv/phase2/results/sinkkv_deterministic_probe | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/ssm_lifecycle/phase0/results/ssml_phase0_20260508T165752Z | 47 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/ssm_shape_codec/phase0/results/ssc_phase0_20260508T173705Z | 47 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| experimental/thoughtflow_fp8/phase2/results/thoughtflow_paper_polish_20260508T0050Z | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/anchor_relative_sparse_packet_gate_20260429_smoke | 14 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/answer_null_predicate_syndrome_20260427 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/arc_controls_20260415 | 1 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/arc_controls_fast_20260415_phase1 | 9 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/arc_controls_fast_20260415_phase1b | 9 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/arc_controls_fast_20260415_phase2 | 3 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/arc_controls_fast_20260415_phase3 | 7 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/arc_heldout_gate_20260415 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/arc_k_only_fixed_20260417 | 8 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/asym_kv_qwen_20260421 | 42 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/asym_kv_real_20260421 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/attention_fidelity_20260418 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/attention_fidelity_gate_20260419 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/attention_match_20260418 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/attention_procrustes_20260418 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/attention_stratified_selector_20260420 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/attention_template_transport_20260418 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/bridge_qkcab_20260419 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/bridge_qkcabbank_20260419 | 2 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/bridge_qkemkd_20260420 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/bridge_ridge_qk_asym_adapter_20260420 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/bridge_ridge_qk_asym_dynmap_adapter_20260420 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
| results/bridge_ridge_qk_asym_predkl_adapter_20260420 | 4 | 0 | 0 | 0 | 0 | quarantined_unimplemented_parser | false | no parser in bounded pass |
