# Kill Board

| method_id | paper | source_path | delta_vs_best_baseline | ci95_high_vs_best_baseline | note |
| --- | --- | --- | --- | --- | --- |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget2.jsonl | -0.4 | -0.27 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget4.jsonl | -0.3090909090909091 | -0.21818181818181817 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget6.jsonl | -0.42342342342342343 | -0.32432432432432434 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget2.jsonl | -0.3125 | -0.1875 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget4.jsonl | -0.4036697247706422 | -0.3211009174311927 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget6.jsonl | -0.3883495145631068 | -0.30097087378640774 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_101/predictions_budget2.jsonl | -0.2653061224489796 | -0.14285714285714285 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_101/predictions_budget4.jsonl | -0.14 | -0.01 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_103/predictions_budget2.jsonl | -0.25688073394495414 | -0.13761467889908258 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_103/predictions_budget4.jsonl | -0.23157894736842105 | -0.11578947368421053 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_107/predictions_budget4.jsonl | -0.2708333333333333 | -0.15625 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_107/predictions_budget6.jsonl | -0.1619047619047619 | -0.0380952380952381 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_fresh_wz_high_entropy | latentwire | results/mac_continue/fresh_mmlu_pro/fresh_mmlu_pro_rows.jsonl | -0.043478260869565216 | 0.2608695652173913 | fresh high-entropy Mac bounded negative |
| L_ORACLE_POWERED_LADDER | latentwire | results/mac_continue/latentwire_oracle_ladder/oracle_ladder_rows.jsonl | -0.013927576601671309 | 0.022284122562674095 | powered dev/gate ladder classifies current score-packet path as not-deployable |
| L_Q1_receiver_query_packet | latentwire | results/mac_continue/latentwire_query_packet/query_packet_rows.jsonl | -0.022284122562674095 | 0.011142061281337047 | final query-conditioned two-way score-packet shot killed; controls do not collapse |
| C2C_candidate_pool_delta_packet | latentwire | results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/candidate_pool_delta_packet_gate.json | -0.218750 |  | matched 3/32 is dominated by coeff-sign-flip control 10/32 |
| C2C_teacher_delta_packet | latentwire | results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/teacher_delta_packet_gate.json | 0.000000 |  | matched 14/32 ties target-only and zero-delta; source-necessary clean count is 0 |
| C2C_generated_answer_packet | latentwire | results/svamp32_c2c_generated_answer_packet_audit_20260505/generated_answer_packet_audit.json | 0.000000 |  | answer value/index equals same-byte visible-answer control; not source-private |
| KVCOMM_existing_cache_smoke_controls | latentwire | results/dense_baseline_mcqa_smoke_20260505/kvcomm_damage_diagnostic_n16.json | 0.000000 | 0.000000 | matched predictions have 1.0 agreement with zero-source controls across inspected smokes |
| C_F_m10_random_control | channel_set | experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z/checker_result.json | -0.761487 |  | random-bin control beats M10; C-F cannot promote before denominator cleanup |
## MPS-First Parked/Negative State

| priority | probe | status | achieved_n | interpretation |
| ---: | --- | --- | ---: | --- |
| 1 | `L_PC1_cross_family_specialist_ceiling` | `PARKED_NEEDS_RECEIVER_CONDITIONED_CEILING_RUNNER` | 768 | No eligible existing artifact measures I(source_signal;Y|receiver_state) for a complementary source-private signal. |
| 2 | `L_PC2_tool_augmented_source_ceiling` | `PARKED_NEEDS_TOOL_PRIVATE_CACHE_AND_RUNNER` | 0 | No dev/gate cache found where the source privately ran calculator/code-exec and receiver did not see the tool result. |
